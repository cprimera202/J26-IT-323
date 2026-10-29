'use client';

import { useState } from 'react';

import Disclaimer from '@/components/Disclaimer';
import ImageDropzone from '@/components/ImageDropzone';
import StatusBadge from '@/components/StatusBadge';
import UploadPanel from '@/components/UploadPanel';
import { ApiError, classifyFundus } from '@/lib/api';
import type { ClassificationResponse } from '@/lib/types';
import { useComponents } from '@/lib/useComponents';
import type { ValidatedUpload } from '@/lib/validateImage';

type Phase = 'idle' | 'ready' | 'running' | 'done' | 'error';

export default function ClassificationPage() {
  const { data, error: statusError } = useComponents();
  const comp = data?.components.find((c) => c.id === 'fundus-classification');
  const modelReady = comp?.status === 'available';

  const [upload, setUpload] = useState<ValidatedUpload | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [result, setResult] = useState<ClassificationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function reset() {
    if (upload?.url) URL.revokeObjectURL(upload.url);
    setUpload(null);
    setResult(null);
    setError(null);
    setPhase('idle');
  }

  async function run() {
    if (!upload) return;
    setPhase('running');
    setError(null);
    setResult(null);
    try {
      setResult(await classifyFundus(upload.file));
      setPhase('done');
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Classification failed.');
      setPhase('error');
    }
  }

  const r = result?.result;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Fundus image classification</h1>
          <p className="mt-1 text-slate-600">Upload a colour fundus photograph to estimate diabetic retinopathy severity (grade 0–4).</p>
        </div>
        <StatusBadge status={comp ? comp.status : statusError ? 'service_unreachable' : 'loading'} />
      </div>
      <Disclaimer />

      {statusError && <p role="alert" className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">{statusError}</p>}
      {comp && !modelReady && (
        <p role="alert" className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">
          The classification model is not available{comp.detail?.error ? `: ${comp.detail.error}` : '.'} Analysis is disabled.
        </p>
      )}

      {!upload ? (
        <ImageDropzone
          label="Upload a fundus image"
          onValid={(u) => {
            setUpload(u);
            setResult(null);
            setError(null);
            setPhase('ready');
          }}
        />
      ) : (
        <UploadPanel upload={upload}>
          <div className="flex flex-wrap gap-2 pt-3">
            <button className="btn-primary" onClick={() => void run()} disabled={!modelReady || phase === 'running'}>
              {phase === 'running' ? 'Analysing…' : 'Run classification'}
            </button>
            <button className="btn-secondary" onClick={reset} disabled={phase === 'running'}>
              Upload another image
            </button>
          </div>
          {phase === 'running' && (
            <p role="status" className="pt-2 text-sm text-slate-500">
              Running the model. This can take several seconds on CPU.
            </p>
          )}
        </UploadPanel>
      )}

      {phase === 'error' && error && (
        <p role="alert" className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">
          {error}
        </p>
      )}

      {result && r && (
        <section className="card space-y-5" aria-labelledby="result-title">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 id="result-title" className="text-sm font-medium uppercase tracking-wide text-slate-500">
                Model output
              </h2>
              <p className="mt-1 text-3xl font-semibold text-slate-900">
                Grade {r.grade}: {r.class_name}
              </p>
              <p className="text-sm text-slate-600">Confidence {(r.confidence * 100).toFixed(1)}%</p>
            </div>
            <div className="text-right text-xs text-slate-500">
              <p>Inference time: {result.inference_ms.toFixed(0)} ms</p>
              <p>
                Image: {result.image.width} × {result.image.height}
              </p>
            </div>
          </div>

          <div
            className={`rounded-lg px-4 py-3 text-sm ${
              r.referable_dr ? 'bg-amber-50 text-amber-900 ring-1 ring-amber-200' : 'bg-emerald-50 text-emerald-900 ring-1 ring-emerald-200'
            }`}
          >
            <p className="font-semibold">
              {r.referable_dr ? 'Meets the configured referable-DR threshold' : 'Does not meet the configured referable-DR threshold'}
            </p>
            <p className="text-xs opacity-80">{r.referable_basis}</p>
          </div>

          {r.low_confidence && (
            <p role="alert" className="rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-900 ring-1 ring-amber-200">
              <strong>Low confidence.</strong> The top class probability is below {(r.low_confidence_threshold * 100).toFixed(0)}%. Interpret this
              result with particular caution.
            </p>
          )}
          {result.quality_notes.map((n) => (
            <p key={n} className="rounded-lg bg-slate-100 px-4 py-2 text-sm text-slate-700">
              {n}
            </p>
          ))}

          <div>
            <h3 className="mb-2 text-sm font-semibold text-slate-800">Class probabilities</h3>
            <ul className="space-y-2">
              {r.probabilities.map((p) => (
                <li key={p.grade} className="grid grid-cols-[8.5rem_1fr_3.5rem] items-center gap-3 text-sm">
                  <span className={p.grade === r.grade ? 'font-semibold text-slate-900' : 'text-slate-600'}>
                    {p.grade} · {p.class_name}
                  </span>
                  <div
                    className="h-2.5 overflow-hidden rounded-full bg-slate-100"
                    role="progressbar"
                    aria-valuenow={Math.round(p.probability * 100)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={`${p.class_name} probability`}
                  >
                    <div className={`h-full ${p.grade === r.grade ? 'bg-brand-600' : 'bg-slate-400'}`} style={{ width: `${p.probability * 100}%` }} />
                  </div>
                  <span className="text-right tabular-nums text-slate-700">{(p.probability * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </div>

          <p className="text-xs text-slate-500">
            This is an automated research prediction, not a confirmed medical diagnosis. Image validation does not establish that disease is present or
            absent.
          </p>
        </section>
      )}
    </div>
  );
}
