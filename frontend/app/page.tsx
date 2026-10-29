'use client';

import Link from 'next/link';

import Disclaimer from '@/components/Disclaimer';
import StatusBadge from '@/components/StatusBadge';
import { LESION_NAMES, LESION_ORDER, type ComponentInfo } from '@/lib/types';
import { useComponents } from '@/lib/useComponents';

const DESCRIPTIONS: Record<string, string> = {
  'fundus-classification': 'Grades diabetic retinopathy severity (0–4) from a colour fundus photograph with class probabilities.',
  'fundus-segmentation': 'Segments microaneurysms, haemorrhages, hard exudates and soft exudates with per-lesion overlays.',
  octa: 'Microvascular analysis, vessel segmentation and DR classification from OCTA scans.',
  oct: 'Retinal layer analysis, fluid / DME analysis and DR-related classification from OCT scans.',
};

const FALLBACK: ComponentInfo[] = [
  { id: 'fundus-classification', name: 'Fundus image classification', status: 'service_unreachable', route: '/classification' },
  { id: 'fundus-segmentation', name: 'Fundus lesion segmentation', status: 'service_unreachable', route: '/segmentation' },
  { id: 'octa', name: 'OCTA analysis', status: 'under_development', route: '/octa' },
  { id: 'oct', name: 'OCT analysis', status: 'under_development', route: '/oct' },
];

export default function Dashboard() {
  const { data, error, loading, refresh } = useComponents();
  const components = data?.components ?? FALLBACK;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-slate-900">Diabetic retinopathy research dashboard</h1>
        <p className="mt-1 text-slate-600">Four analysis components. Availability below reflects the models that are actually loaded.</p>
      </div>
      <Disclaimer />

      {error && (
        <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">
          <span>{error} Start the Node backend and the Python inference service, then retry.</span>
          <button className="btn-secondary" onClick={() => void refresh()}>
            Retry
          </button>
        </div>
      )}
      {data && !data.inferenceService.reachable && (
        <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">
          The Python inference service is not reachable, so no model can run.
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        {components.map((c) => {
          const dev = c.status === 'under_development';
          const usable = c.status === 'available' || c.status === 'partial';
          return (
            <section key={c.id} className="card flex flex-col" aria-labelledby={`${c.id}-title`}>
              <div className="flex items-start justify-between gap-3">
                <h2 id={`${c.id}-title`} className="text-lg font-semibold text-slate-900">
                  {c.name}
                </h2>
                <StatusBadge status={loading && !data ? 'loading' : c.status} />
              </div>
              <p className="mt-2 text-sm text-slate-600">{DESCRIPTIONS[c.id]}</p>

              {c.id === 'fundus-segmentation' && c.lesions && (
                <ul className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs" aria-label="Lesion model status">
                  {LESION_ORDER.map((code) => {
                    const l = c.lesions?.[code];
                    return (
                      <li key={code} className="flex items-center gap-1.5">
                        <span aria-hidden className={`h-2 w-2 rounded-full ${l?.ready ? 'bg-emerald-500' : 'bg-red-500'}`} />
                        <span className="text-slate-700">{LESION_NAMES[code]}</span>
                        <span className="text-slate-500">{l?.ready ? 'ready' : 'unavailable'}</span>
                      </li>
                    );
                  })}
                </ul>
              )}
              {c.id === 'fundus-classification' && c.detail && !c.detail.ready && (
                <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-xs text-red-800">{c.detail.error}</p>
              )}
              {dev && c.planned && (
                <ul className="mt-3 list-inside list-disc text-xs text-slate-500">
                  {c.planned.map((p) => (
                    <li key={p}>Planned: {p}</li>
                  ))}
                </ul>
              )}

              <div className="mt-auto pt-4">
                <Link href={c.route} className={usable ? 'btn-primary' : 'btn-secondary'}>
                  {dev ? 'View (not available yet)' : usable ? 'Open' : 'Open (model unavailable)'}
                </Link>
              </div>
            </section>
          );
        })}
      </div>

      <section className="card">
        <h2 className="text-base font-semibold text-slate-900">Recent analyses</h2>
        <p className="mt-1 text-sm text-slate-500">
          No analysis history is kept. Uploaded images and results are processed in memory and discarded, so there is nothing to list here.
        </p>
      </section>
    </div>
  );
}
