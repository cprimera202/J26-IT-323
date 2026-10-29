import cors from 'cors';
import express from 'express';
import multer from 'multer';

import { loadConfig } from './config.js';
import { createInferenceClient, InferenceError } from './inferenceClient.js';
import { providers, UNDER_DEVELOPMENT } from './providers/index.js';
import { parseLesions, sniffImage } from './validate.js';

export const DISCLAIMER =
  'Research use only. Not a medical device and not for standalone clinical diagnosis.';

const sendError = (res, status, code, message, extra = {}) =>
  res.status(status).json({ error: { code, message }, ...extra });

export function createApp(config = loadConfig(), client = createInferenceClient(config)) {
  const app = express();
  app.disable('x-powered-by');

  app.use(
    cors({
      origin(origin, cb) {
        // Non-browser callers (no Origin header) are allowed; browsers must be on the allow-list.
        if (!origin || config.allowedOrigins.includes(origin)) return cb(null, true);
        return cb(null, false);
      },
      methods: ['GET', 'POST'],
    }),
  );
  app.use((req, res, next) => {
    res.set('X-Content-Type-Options', 'nosniff');
    res.set('Cache-Control', 'no-store');
    next();
  });

  const uploader = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: config.maxUploadBytes, files: 1, fields: 4, parts: 6 },
  }).single('file');

  function uploadImage(req, res, next) {
    uploader(req, res, (err) => {
      if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
          return sendError(res, 413, 'file_too_large', `File exceeds the ${Math.round(config.maxUploadBytes / 1048576)} MB limit.`);
        }
        return sendError(res, 400, 'invalid_upload', 'Invalid upload. Send one image in the "file" field.');
      }
      if (err) return sendError(res, 400, 'invalid_upload', 'The upload could not be processed.');
      if (!req.file) return sendError(res, 400, 'missing_file', 'No image was uploaded. Use the "file" field.');
      if (req.file.size === 0) return sendError(res, 400, 'empty_file', 'The uploaded file is empty.');
      const sniffed = sniffImage(req.file.buffer);
      if (!sniffed) return sendError(res, 415, 'unsupported_format', 'Only JPEG and PNG images are supported.');
      req.sniffed = sniffed;
      next();
    });
  }

  const wrap = (fn) => (req, res, next) => fn(req, res).catch(next);

  // ------------------------------------------------------------------ health
  app.get(
    '/api/health',
    wrap(async (_req, res) => {
      let inference = { reachable: false, modelsReady: null };
      try {
        const h = await client.health();
        inference.reachable = h.status === 200;
      } catch {
        /* unreachable */
      }
      if (inference.reachable) {
        try {
          const m = await client.models();
          inference.modelsReady = `${m.body.ready_count}/${m.body.total_count}`;
          inference.allModelsReady = Boolean(m.body.all_ready);
        } catch {
          /* ignore */
        }
      }
      res.json({ status: 'ok', service: 'dr-backend', inference });
    }),
  );

  // ------------------------------------------------------------------ availability
  app.get(
    '/api/components',
    wrap(async (_req, res) => {
      let models = null;
      let unreachable = false;
      try {
        const r = await client.models();
        if (r.status === 200) models = r.body;
      } catch {
        unreachable = true;
      }

      const cls = models?.classification;
      const segEntries = models ? Object.values(models.segmentation) : [];
      const segReady = segEntries.filter((s) => s.ready).length;
      const state = (ready, total) => {
        if (unreachable || !models) return 'service_unreachable';
        if (ready === total) return 'available';
        return ready === 0 ? 'unavailable' : 'partial';
      };

      res.json({
        inferenceService: { reachable: !unreachable && Boolean(models), device: models?.device ?? null },
        components: [
          {
            id: 'fundus-classification',
            name: 'Fundus image classification',
            status: state(cls?.ready ? 1 : 0, 1),
            route: '/classification',
            detail: cls ? { ready: cls.ready, error: cls.error, architecture: cls.architecture, classNames: cls.class_names } : null,
          },
          {
            id: 'fundus-segmentation',
            name: 'Fundus lesion segmentation',
            status: state(segReady, 4),
            route: '/segmentation',
            readyCount: segReady,
            totalCount: 4,
            lesions: models ? models.segmentation : null,
          },
          { ...UNDER_DEVELOPMENT.octa, status: 'under_development', route: '/octa', modelReady: false },
          { ...UNDER_DEVELOPMENT.oct, status: 'under_development', route: '/oct', modelReady: false },
        ],
        disclaimer: DISCLAIMER,
      });
    }),
  );

  // ------------------------------------------------------------------ fundus
  app.post(
    '/api/fundus/classify',
    uploadImage,
    wrap(async (req, res) => {
      res.json(await client.classify(req.file, req.sniffed));
    }),
  );

  app.post(
    '/api/fundus/segment',
    uploadImage,
    wrap(async (req, res) => {
      const lesions = parseLesions(req.query.lesions);
      if (lesions === null) return sendError(res, 422, 'invalid_lesion', 'lesions must be a comma list of MA, HE, EX, SE.');
      res.json(await client.segment(req.file, req.sniffed, lesions));
    }),
  );

  // ------------------------------------------------------------------ OCTA / OCT (no model yet)
  for (const key of ['octa', 'oct']) {
    app.post(
      `/api/${key}/analyze`,
      uploadImage,
      wrap(async (req, res) => {
        const provider = providers[key];
        if (!provider) {
          return sendError(
            res,
            501,
            'model_not_available',
            `${UNDER_DEVELOPMENT[key].name} is under development. No trained model is available, so no analysis was run.`,
          );
        }
        res.json(await provider.analyze(req.file, req.sniffed, client));
      }),
    );
  }

  // ------------------------------------------------------------------ errors
  app.use('/api', (_req, res) => sendError(res, 404, 'not_found', 'Unknown API route.'));

  // eslint-disable-next-line no-unused-vars
  app.use((err, _req, res, _next) => {
    if (err instanceof InferenceError) return sendError(res, err.status, err.code, err.message, err.extra?.lesions ? { lesions: err.extra.lesions } : {});
    console.error('Unhandled error:', err?.name); // never log request bodies or file contents
    return sendError(res, 500, 'internal_error', 'An internal error occurred.');
  });

  return app;
}
