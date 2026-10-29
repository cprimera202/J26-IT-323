'use client';

import { useState } from 'react';

import Disclaimer from './Disclaimer';
import ImageDropzone from './ImageDropzone';
import StatusBadge from './StatusBadge';
import UploadPanel from './UploadPanel';
import type { ValidatedUpload } from '@/lib/validateImage';

interface Props {
  title: string;
  modality: 'OCTA' | 'OCT';
  intro: string;
  planned: { title: string; text: string }[];
}

/**
 * Shared shell for modalities whose models do not exist yet. It validates and previews an upload but
 * never calls the backend and never shows analysis output. When a real model is connected, replace the
 * disabled button with a call through lib/api.ts (routes /api/octa/analyze and /api/oct/analyze exist).
 */
export default function ModalityPage({ title, modality, intro, planned }: Props) {
  const [upload, setUpload] = useState<ValidatedUpload | null>(null);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">{title}</h1>
          <p className="mt-1 text-slate-600">{intro}</p>
        </div>
        <StatusBadge status="under_development" />
      </div>
      <Disclaimer />

      <div role="status" className="rounded-xl border border-slate-300 bg-slate-100 px-5 py-4">
        <p className="font-semibold text-slate-900">Under Development — Model Not Available</p>
        <p className="mt-1 text-sm text-slate-700">
          No trained {modality} model is connected yet, so no analysis can be run. This page shows no predictions, measurements or classifications.
          You can check that an image uploads and previews correctly.
        </p>
      </div>

      {!upload ? (
        <ImageDropzone label={`Upload an ${modality} image (preview only)`} onValid={setUpload} />
      ) : (
        <UploadPanel upload={upload}>
          <div className="flex flex-wrap gap-2 pt-3">
            <button className="btn-primary" disabled aria-disabled="true" title="Analysis is unavailable: no trained model">
              Run analysis (unavailable)
            </button>
            <button className="btn-secondary" onClick={() => setUpload(null)}>
              Remove image
            </button>
          </div>
          <p className="pt-2 text-sm text-slate-500">Analysis status: not available — model not implemented.</p>
        </UploadPanel>
      )}

      <section className="card">
        <h2 className="text-base font-semibold text-slate-900">Planned functionality</h2>
        <p className="mt-1 text-sm text-slate-500">These capabilities are not implemented and produce no output today.</p>
        <ul className="mt-3 grid gap-3 sm:grid-cols-3">
          {planned.map((p) => (
            <li key={p.title} className="rounded-lg border border-dashed border-slate-300 p-3">
              <p className="text-sm font-medium text-slate-800">{p.title}</p>
              <p className="mt-1 text-xs text-slate-500">{p.text}</p>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
