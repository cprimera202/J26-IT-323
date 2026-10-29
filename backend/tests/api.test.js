import assert from 'node:assert/strict';
import http from 'node:http';
import { after, before, describe, test } from 'node:test';

import { createApp } from '../src/app.js';
import { loadConfig } from '../src/config.js';

const PNG = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.alloc(64, 1)]);
const JPG = Buffer.concat([Buffer.from([0xff, 0xd8, 0xff, 0xe0]), Buffer.alloc(64, 2)]);

const MODELS = {
  device: 'cpu',
  ready_count: 3,
  total_count: 5,
  all_ready: false,
  classification: { ready: true, error: null, architecture: 'resnet50', class_names: ['a', 'b', 'c', 'd', 'e'] },
  segmentation: {
    MA: { ready: true, status: 'ready', threshold: 0.5 },
    HE: { ready: true, status: 'ready', threshold: 0.55 },
    EX: { ready: false, status: 'unavailable', error: 'Checkpoint does not match' },
    SE: { ready: false, status: 'unavailable', error: 'Checkpoint not found' },
  },
};

let fake; // fake Python service
let fakePort;
const seen = [];
let mode = 'ok'; // ok | slow | error503

function listen(server) {
  return new Promise((r) => server.listen(0, '127.0.0.1', () => r(server.address().port)));
}

async function startBackend(overrides = {}) {
  const config = loadConfig({
    inferenceBaseUrl: `http://127.0.0.1:${fakePort}`,
    inferenceTimeoutMs: 800,
    maxUploadBytes: 1024 * 1024,
    allowedOrigins: ['http://localhost:3000'],
    ...overrides,
  });
  const server = http.createServer(createApp(config));
  const port = await listen(server);
  return { server, url: `http://127.0.0.1:${port}` };
}

function form(buf, name = 'x.png', type = 'image/png') {
  const f = new FormData();
  f.append('file', new Blob([buf], { type }), name);
  return f;
}

before(async () => {
  fake = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const body = Buffer.concat(chunks);
      seen.push({ url: req.url, method: req.method, contentType: req.headers['content-type'], body });
      const json = (code, obj) => {
        res.writeHead(code, { 'content-type': 'application/json' });
        res.end(JSON.stringify(obj));
      };
      if (req.url === '/health') return json(200, { status: 'ok' });
      if (req.url === '/models') return json(200, MODELS);
      if (mode === 'slow') return setTimeout(() => json(200, {}), 3000);
      if (mode === 'error503') return json(503, { error: { code: 'model_unavailable', message: "Model 'classification' is not available: x" } });
      if (req.url === '/v1/fundus/classify') return json(200, { result: { grade: 2, class_name: 'Moderate DR' }, inference_ms: 5 });
      if (req.url.startsWith('/v1/fundus/segment')) return json(200, { lesions: { MA: { status: 'ok' } } });
      return json(404, {});
    });
  });
  fakePort = await listen(fake);
});
after(() => fake.close());

describe('backend', () => {
  let b;
  before(async () => (b = await startBackend()));
  after(() => b.server.close());

  test('GET /api/health reports node + inference reachability', async () => {
    const r = await fetch(`${b.url}/api/health`);
    const j = await r.json();
    assert.equal(r.status, 200);
    assert.equal(j.status, 'ok');
    assert.equal(j.inference.reachable, true);
    assert.equal(j.inference.modelsReady, '3/5');
  });

  test('GET /api/components reports real availability and OCT/OCTA as under development', async () => {
    const j = await (await fetch(`${b.url}/api/components`)).json();
    const by = Object.fromEntries(j.components.map((c) => [c.id, c]));
    assert.equal(by['fundus-classification'].status, 'available');
    assert.equal(by['fundus-segmentation'].status, 'partial');
    assert.equal(by['fundus-segmentation'].readyCount, 2);
    assert.equal(by.octa.status, 'under_development');
    assert.equal(by.oct.status, 'under_development');
    assert.match(j.disclaimer, /Research use only/);
  });

  test('classify forwards a valid image as multipart and returns the real response', async () => {
    mode = 'ok';
    seen.length = 0;
    const r = await fetch(`${b.url}/api/fundus/classify`, { method: 'POST', body: form(PNG, '../../evil name.png') });
    assert.equal(r.status, 200);
    assert.equal((await r.json()).result.grade, 2);
    const fwd = seen.find((s) => s.url === '/v1/fundus/classify');
    assert.match(fwd.contentType, /multipart\/form-data/);
    const text = fwd.body.toString('latin1');
    assert.ok(text.includes('filename="upload.png"'), 'server-chosen filename must be used');
    assert.ok(!text.includes('evil'), 'client filename must never be forwarded');
  });

  test('segment accepts JPEG and forwards a normalised lesions filter', async () => {
    seen.length = 0;
    const r = await fetch(`${b.url}/api/fundus/segment?lesions=ma,he`, { method: 'POST', body: form(JPG, 'a.jpg', 'image/jpeg') });
    assert.equal(r.status, 200);
    assert.ok(seen.some((s) => s.url === '/v1/fundus/segment?lesions=MA%2CHE'));
  });

  test('segment rejects invalid lesion filter', async () => {
    for (const q of ['XX', '../etc', 'MA;HE']) {
      const r = await fetch(`${b.url}/api/fundus/segment?lesions=${encodeURIComponent(q)}`, { method: 'POST', body: form(PNG) });
      assert.equal(r.status, 422, q);
    }
  });

  test('rejects non-image content even with an image extension/MIME', async () => {
    const r = await fetch(`${b.url}/api/fundus/classify`, { method: 'POST', body: form(Buffer.from('<html>not an image</html>'), 'a.png') });
    assert.equal(r.status, 415);
    assert.equal((await r.json()).error.code, 'unsupported_format');
  });

  test('rejects missing and empty uploads', async () => {
    const none = await fetch(`${b.url}/api/fundus/classify`, { method: 'POST', body: new FormData() });
    assert.equal(none.status, 400);
    const empty = await fetch(`${b.url}/api/fundus/classify`, { method: 'POST', body: form(Buffer.alloc(0)) });
    assert.equal(empty.status, 400);
  });

  test('rejects oversized uploads with 413', async () => {
    const big = Buffer.concat([PNG, Buffer.alloc(1024 * 1024 + 10)]);
    const r = await fetch(`${b.url}/api/fundus/classify`, { method: 'POST', body: form(big) });
    assert.equal(r.status, 413);
    assert.equal((await r.json()).error.code, 'file_too_large');
  });

  test('OCT and OCTA return 501 model_not_available and never fabricate results', async () => {
    for (const key of ['octa', 'oct']) {
      const r = await fetch(`${b.url}/api/${key}/analyze`, { method: 'POST', body: form(PNG) });
      const j = await r.json();
      assert.equal(r.status, 501);
      assert.equal(j.error.code, 'model_not_available');
      assert.equal(j.result, undefined);
    }
  });

  test('passes through model_unavailable (503) from the inference service', async () => {
    mode = 'error503';
    const r = await fetch(`${b.url}/api/fundus/classify`, { method: 'POST', body: form(PNG) });
    mode = 'ok';
    assert.equal(r.status, 503);
    assert.equal((await r.json()).error.code, 'model_unavailable');
  });

  test('times out slow inference with 504', async () => {
    mode = 'slow';
    const r = await fetch(`${b.url}/api/fundus/classify`, { method: 'POST', body: form(PNG) });
    mode = 'ok';
    assert.equal(r.status, 504);
    assert.equal((await r.json()).error.code, 'inference_timeout');
  });

  test('CORS allows only configured origins', async () => {
    const ok = await fetch(`${b.url}/api/health`, { headers: { Origin: 'http://localhost:3000' } });
    assert.equal(ok.headers.get('access-control-allow-origin'), 'http://localhost:3000');
    const bad = await fetch(`${b.url}/api/health`, { headers: { Origin: 'https://evil.example' } });
    assert.equal(bad.headers.get('access-control-allow-origin'), null);
  });

  test('unknown API routes return JSON 404', async () => {
    const r = await fetch(`${b.url}/api/nope`);
    assert.equal(r.status, 404);
    assert.equal((await r.json()).error.code, 'not_found');
  });
});

describe('backend with inference service down', () => {
  let b;
  before(async () => (b = await startBackend({ inferenceBaseUrl: 'http://127.0.0.1:1' })));
  after(() => b.server.close());

  test('classify returns 503 inference_unreachable without leaking internals', async () => {
    const r = await fetch(`${b.url}/api/fundus/classify`, { method: 'POST', body: form(PNG) });
    const text = await r.text();
    assert.equal(r.status, 503);
    assert.ok(text.includes('inference_unreachable'));
    assert.ok(!text.includes('127.0.0.1') && !text.includes('ECONNREFUSED'));
  });

  test('components reports service_unreachable (not "available")', async () => {
    const j = await (await fetch(`${b.url}/api/components`)).json();
    assert.equal(j.inferenceService.reachable, false);
    assert.equal(j.components[0].status, 'service_unreachable');
    assert.equal(j.components[1].status, 'service_unreachable');
  });

  test('health stays up and reports inference unreachable', async () => {
    const j = await (await fetch(`${b.url}/api/health`)).json();
    assert.equal(j.status, 'ok');
    assert.equal(j.inference.reachable, false);
  });
});
