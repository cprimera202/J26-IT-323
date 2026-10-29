'use client';

import { useState } from 'react';

import Disclaimer from '@/components/Disclaimer';
import ImageDropzone from '@/components/ImageDropzone';
import SegmentationViewer from '@/components/SegmentationViewer';
import StatusBadge from '@/components/StatusBadge';
import UploadPanel from '@/components/UploadPanel';
import { ApiError, segmentFundus } from '@/lib/api';
import { LESION_NAMES, LESION_ORDER, type SegmentationResponse } from '@/lib/types';
import { useComponents } from '@/lib/useComponents';
import type { ValidatedUpload } from '@/lib/validateImage';

export default function SegmentationPage() {
  const { data: comps, error: statusError } = useComponents();
  const comp = comps?.components.find((c) => c.id === 'fundus-segmentation');
  const anyReady = (comp?.readyCount ?? 0) > 0;

  const [upload, setUpload] = useState<ValidatedUpload | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SegmentationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function reset() {
    if (upload?.url) URL.revokeObjectURL(upload.url);
    setUpload(null);
    setResult(null);
    setError(null);
  }

  async function run() {
    if (!upload) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      setResult(await segmentFundus(upload.file));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Segmentation failed.');
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Fundus lesion segmentation</h1>
          <p className="mt-1 text-slate-600">
            Four independent models: microaneurysms, haemorrhages, hard exudates and soft exudates.
            {comp ? ` ${comp.readyCount}/${comp.totalCount} models are loaded.` : ''}
          </p>
        </div>
        <StatusBadge status={comp ? comp.status : statusError ? 'service_unreachable' : 'loading'} />
      </div>
      <Disclaimer />

      {statusError && <p role="alert" className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">{statusError}</p>}

      {comp?.lesions && (
        <div className="card">
          <h2 className="mb-2 text-sm font-semibold text-slate-800">Model status</h2>
          <ul className="grid gap-2 text-sm sm:grid-cols-2">
            {LESION_ORDER.map((code) => {
              const l = comp.lesions?.[code];
              return (
                <li key={code} className="rounded-lg bg-slate-50 px-3 py-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{LESION_NAMES[code]}</span>
                    <span className={l?.ready ? 'text-emerald-700' : 'text-red-700'}>{l?.ready ? 'Ready' : 'Unavailable'}</span>
                  </div>
                  <div className="text-xs text-slate-500">
                    Threshold {l?.threshold ?? '—'}
                    {l?.architecture ? ` · ${l.architecture}` : ''}
                  </div>
                  {!l?.ready && l?.error && <div className="mt-1 text-xs text-red-700">{l.error}</div>}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {!upload ? (
        <ImageDropzone
          label="Upload a fundus image"
          onValid={(u) => {
            setUpload(u);
            setResult(null);
            setError(null);
          }}
        />
      ) : (
        <UploadPanel upload={upload}>
          <div className="flex flex-wrap gap-2 pt-3">
            <button className="btn-primary" onClick={() => void run()} disabled={!anyReady || running}>
              {running ? 'Segmenting…' : 'Run segmentation'}
            </button>
            <button className="btn-secondary" onClick={reset} disabled={running}>
              Upload another image
            </button>
          </div>
          {running && <p role="status" className="pt-2 text-sm text-slate-500">Running up to four models. This can take a while on CPU.</p>}
          {!anyReady && comp && <p className="pt-2 text-sm text-red-700">No segmentation model is available, so analysis is disabled.</p>}
        </UploadPanel>
      )}

      {error && <p role="alert" className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">{error}</p>}

      {result && upload && (
        <>
          {result.quality_notes.map((n) => (
            <p key={n} className="rounded-lg bg-slate-100 px-4 py-2 text-sm text-slate-700">{n}</p>
          ))}
          <SegmentationViewer imageUrl={upload.url} data={result} file={upload.file} />
          <p className="text-xs text-slate-500">
            Masks are automated research predictions and have not been clinically validated. They do not confirm the presence or absence of disease.
          </p>
        </>
      )}
    </div>
  );
}
