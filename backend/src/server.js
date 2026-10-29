import { createApp } from './app.js';
import { loadConfig } from './config.js';

const config = loadConfig();
const app = createApp(config);

const server = app.listen(config.port, () => {
  console.log(`DR backend listening on http://127.0.0.1:${config.port}`);
  console.log(`Inference service: ${config.inferenceBaseUrl}`);
  console.log(`Allowed frontend origins: ${config.allowedOrigins.join(', ')}`);
});

for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => server.close(() => process.exit(0)));
