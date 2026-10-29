import type { ClassificationResponse, ComponentsResponse, SegmentationResponse } from './types';

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:5000').replace(/\/+$/, '');

/** Every backend route the frontend calls. Kept in one place so a test can compare it with the backend. */
export const ROUTES = {
  health: '/api/health',
  components: '/api/components',
  classify: '/api/fundus/classify',
  segment: '/api/fundus/segment',
  octaAnalyze: '/api/octa/analyze',
  octAnalyze: '/api/oct/analyze',
} as const;

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status = 0,
    public lesions?: Record<string, { status: string; error?: string }>,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

const REQUEST_TIMEOUT_MS = 150_000;

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, signal: ctrl.signal });
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') {
      throw new ApiError('timeout', 'The request timed out. Please try again.');
    }
    throw new ApiError('network', 'Cannot reach the backend. Check that the Node server is running.');
  } finally {
    clearTimeout(timer);
  }
  let body: any = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    const e = body?.error;
    throw new ApiError(e?.code ?? 'request_failed', e?.message ?? `Request failed (HTTP ${res.status}).`, res.status, body?.lesions);
  }
  if (body === null) throw new ApiError('bad_response', 'The backend returned an unreadable response.', res.status);
  return body as T;
}

function uploadBody(file: File): FormData {
  const form = new FormData();
  form.append('file', file);
  return form;
}

export const getComponents = () => request<ComponentsResponse>(ROUTES.components, {}, 15_000);

export const classifyFundus = (file: File) =>
  request<ClassificationResponse>(ROUTES.classify, { method: 'POST', body: uploadBody(file) });

export const segmentFundus = (file: File, lesions?: string[]) =>
  request<SegmentationResponse>(
    `${ROUTES.segment}${lesions?.length ? `?lesions=${encodeURIComponent(lesions.join(','))}` : ''}`,
    { method: 'POST', body: uploadBody(file) },
  );
