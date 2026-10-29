const int = (name, fallback) => {
  const v = Number.parseInt(process.env[name] ?? '', 10);
  return Number.isFinite(v) && v > 0 ? v : fallback;
};

export function loadConfig(overrides = {}) {
  return {
    port: int('PORT', 5000),
    inferenceBaseUrl: (process.env.INFERENCE_BASE_URL || 'http://127.0.0.1:8001').replace(/\/+$/, ''),
    inferenceTimeoutMs: int('INFERENCE_TIMEOUT_MS', 120_000),
    healthTimeoutMs: int('HEALTH_TIMEOUT_MS', 4_000),
    maxUploadBytes: int('MAX_UPLOAD_MB', 10) * 1024 * 1024,
    allowedOrigins: (process.env.FRONTEND_ORIGINS || 'http://localhost:3000,http://127.0.0.1:3000')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),
    ...overrides,
  };
}
