import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError, classifyFundus, getComponents, segmentFundus } from '@/lib/api';

const file = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])], 'a.png', { type: 'image/png' });

function mockFetch(status: number, body: unknown, isJson = true) {
  const fn = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: isJson ? async () => body : async () => { throw new Error('not json'); },
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe('api client', () => {
  it('returns the parsed classification response on success', async () => {
    const fn = mockFetch(200, { result: { grade: 2 }, inference_ms: 3 });
    const out = await classifyFundus(file);
    expect(out.result.grade).toBe(2);
    const [url, init] = fn.mock.calls[0];
    expect(url).toMatch(/\/api\/fundus\/classify$/);
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
  });

  it('sends the lesion filter for segmentation', async () => {
    const fn = mockFetch(200, { lesions: {} });
    await segmentFundus(file, ['MA', 'HE']);
    expect(fn.mock.calls[0][0]).toMatch(/\/api\/fundus\/segment\?lesions=MA%2CHE$/);
  });

  it('surfaces backend error code, message and status', async () => {
    mockFetch(503, { error: { code: 'model_unavailable', message: 'Model not available' } });
    await expect(classifyFundus(file)).rejects.toMatchObject({ code: 'model_unavailable', message: 'Model not available', status: 503 });
  });

  it('keeps per-lesion details of a 503 segmentation error', async () => {
    mockFetch(503, { error: { code: 'model_unavailable', message: 'None available' }, lesions: { EX: { status: 'unavailable', error: 'x' } } });
    const err = await segmentFundus(file).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.lesions.EX.status).toBe('unavailable');
  });

  it('maps network failures to a friendly error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await expect(getComponents()).rejects.toMatchObject({ code: 'network' });
  });

  it('handles non-JSON error bodies', async () => {
    mockFetch(502, null, false);
    await expect(classifyFundus(file)).rejects.toMatchObject({ code: 'request_failed', status: 502 });
  });

  it('never fabricates a result for an unreadable success body', async () => {
    mockFetch(200, null, false);
    await expect(classifyFundus(file)).rejects.toMatchObject({ code: 'bad_response' });
  });
});
