// Thin client for the Python inference service. All failures become structured errors.
export class InferenceError extends Error {
  constructor(status, code, message, extra = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.extra = extra;
  }
}

async function parseJson(res) {
  try {
    return await res.json();
  } catch {
    throw new InferenceError(502, 'bad_gateway', 'The inference service returned an unreadable response.');
  }
}

function wrapNetworkError(err) {
  if (err instanceof InferenceError) return err;
  if (err?.name === 'TimeoutError' || err?.name === 'AbortError') {
    return new InferenceError(504, 'inference_timeout', 'The inference service took too long to respond.');
  }
  return new InferenceError(503, 'inference_unreachable', 'The inference service is not reachable. Make sure it is running.');
}

export function createInferenceClient(config) {
  const base = config.inferenceBaseUrl;

  async function getJson(path, timeoutMs) {
    try {
      const res = await fetch(`${base}${path}`, { signal: AbortSignal.timeout(timeoutMs) });
      return { status: res.status, body: await parseJson(res) };
    } catch (err) {
      throw wrapNetworkError(err);
    }
  }

  async function postImage(path, file, sniffed, query = '') {
    const form = new FormData();
    // Fixed server-chosen filename: user-supplied names are never forwarded.
    form.append('file', new Blob([file.buffer], { type: sniffed.mime }), `upload.${sniffed.ext}`);
    let res;
    let body;
    try {
      res = await fetch(`${base}${path}${query ? `?${query}` : ''}`, {
        method: 'POST',
        body: form,
        signal: AbortSignal.timeout(config.inferenceTimeoutMs),
      });
      body = await parseJson(res);
    } catch (err) {
      throw wrapNetworkError(err);
    }
    if (!res.ok) {
      const e = body?.error ?? {};
      // Pass through known client/availability errors; hide anything else behind a generic 502.
      const status = [400, 413, 415, 422, 503].includes(res.status) ? res.status : 502;
      throw new InferenceError(status, e.code || 'inference_failed', e.message || 'Inference failed.', {
        lesions: body?.lesions,
      });
    }
    return body;
  }

  return {
    health: () => getJson('/health', config.healthTimeoutMs),
    models: () => getJson('/models', config.healthTimeoutMs),
    classify: (file, sniffed) => postImage('/v1/fundus/classify', file, sniffed),
    segment: (file, sniffed, lesions) =>
      postImage('/v1/fundus/segment', file, sniffed, lesions ? `lesions=${encodeURIComponent(lesions)}` : ''),
  };
}
