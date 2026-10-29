import { readFileSync } from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { ROUTES } from '@/lib/api';

const root = path.resolve(__dirname, '..', '..');
const read = (p: string) => readFileSync(path.join(root, p), 'utf8');

describe('route consistency: frontend ↔ Node backend ↔ FastAPI', () => {
  const backend = read('backend/src/app.js');
  const client = read('backend/src/inferenceClient.js');
  const python = read('inference_service/app/main.py');

  it('every frontend route is served by the Node backend', () => {
    for (const route of Object.values(ROUTES)) {
      const dynamic = route.replace(/\/(octa|oct)\/analyze$/, '/${key}/analyze');
      expect(backend.includes(`'${route}'`) || backend.includes(`\`${dynamic}\``), route).toBe(true);
    }
  });

  it('every inference path called by the Node backend exists in FastAPI', () => {
    const paths = [...client.matchAll(/'(\/(?:v1\/[a-z/]+|health|models))'/g)].map((m) => m[1]);
    expect(paths.length).toBeGreaterThanOrEqual(4);
    for (const p of new Set(paths)) expect(python.includes(`"${p}"`), p).toBe(true);
  });
});
