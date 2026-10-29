'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

import { base64ToBlob, formatPercent, maskToBinaryRgba, maskToRgba, rgbCss } from '@/lib/overlay';
import { LESION_COLORS, LESION_NAMES, LESION_ORDER, type LesionResult, type SegmentationResponse } from '@/lib/types';

type MaskCanvas = { tinted: HTMLCanvasElement; binary: HTMLCanvasElement };

function loadImage(src: string, label = 'image'): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not decode ${label}`));
    img.src = src;
  });
}

function makeCanvas(w: number, h: number, rgba: Uint8ClampedArray): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(w, h);
  img.data.set(rgba);
  ctx.putImageData(img, 0, 0);
  return c;
}

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function SegmentationViewer({
  imageUrl,
  data,
  file,
}: {
  imageUrl: string;
  data: SegmentationResponse;
  file?: File;
}) {
  const { width: W, height: H } = data.image;
  const okCodes = useMemo(() => LESION_ORDER.filter((c) => data.lesions[c]?.status === 'ok'), [data]);

  const [original, setOriginal] = useState<HTMLImageElement | null>(null);
  const [masks, setMasks] = useState<Record<string, MaskCanvas>>({});
  const [visible, setVisible] = useState<Record<string, boolean>>({});
  const [opacity, setOpacity] = useState(0.6);
  const [view, setView] = useState<'overlay' | string>('overlay'); // 'overlay' or a lesion code (mask view)
  const [loadError, setLoadError] = useState<string | null>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<HTMLCanvasElement>(null);

  // Decode the original image and every returned mask at their native (original-image) resolution.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        let orig: HTMLImageElement;
        try {
          orig = await loadImage(imageUrl, 'original image');
        } catch (origErr) {
          if (file) {
            const fallbackUrl = URL.createObjectURL(file);
            try {
              orig = await loadImage(fallbackUrl, 'original image');
            } finally {
              URL.revokeObjectURL(fallbackUrl);
            }
          } else {
            throw origErr;
          }
        }

        const next: Record<string, MaskCanvas> = {};
        for (const code of okCodes) {
          const les = data.lesions[code];
          if (!les?.mask_png_base64) continue;
          const blob = base64ToBlob(les.mask_png_base64);
          if (blob.size === 0) continue;
          const url = URL.createObjectURL(blob);
          try {
            const m = await loadImage(url, `${code} mask`);
            if (m.naturalWidth !== W || m.naturalHeight !== H) {
              throw new Error(`${code} mask size ${m.naturalWidth}×${m.naturalHeight} does not match the image ${W}×${H}`);
            }
            const tmp = document.createElement('canvas');
            tmp.width = W;
            tmp.height = H;
            const ctx = tmp.getContext('2d')!;
            ctx.drawImage(m, 0, 0);
            const px = ctx.getImageData(0, 0, W, H).data;
            next[code] = {
              tinted: makeCanvas(W, H, maskToRgba(px, LESION_COLORS[code])),
              binary: makeCanvas(W, H, maskToBinaryRgba(px)),
            };
          } finally {
            URL.revokeObjectURL(url);
          }
        }
        if (cancelled) return;
        setOriginal(orig);
        setMasks(next);
        setVisible(Object.fromEntries(okCodes.map((c) => [c, true])));
        setLoadError(null);
      } catch (e) {
        if (!cancelled) setLoadError((e as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [imageUrl, file, data, okCodes, W, H]);

  const drawOverlay = (target: HTMLCanvasElement, withImage = true) => {
    if (!original) return;
    target.width = W;
    target.height = H;
    const ctx = target.getContext('2d')!;
    if (withImage) ctx.drawImage(original, 0, 0, W, H);
    ctx.globalAlpha = opacity;
    for (const code of okCodes) if (visible[code] && masks[code]) ctx.drawImage(masks[code].tinted, 0, 0);
    ctx.globalAlpha = 1;
  };

  useEffect(() => {
    if (overlayRef.current) drawOverlay(overlayRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [original, masks, visible, opacity]);

  useEffect(() => {
    const c = maskRef.current;
    if (!c || !original) return;
    c.width = W;
    c.height = H;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);
    if (view !== 'overlay' && masks[view]) ctx.drawImage(masks[view].binary, 0, 0);
  }, [original, masks, view, W, H]);

  const anyVisible = okCodes.some((c) => visible[c]);

  function saveOverlay() {
    const c = document.createElement('canvas');
    drawOverlay(c);
    c.toBlob((b) => b && download(b, 'segmentation_overlay.png'), 'image/png');
  }

  return (
    <section className="space-y-5" aria-label="Segmentation results">
      {loadError && (
        <p role="alert" className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">
          Could not display the segmentation result: {loadError}
        </p>
      )}

      <div className="card space-y-4">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <fieldset className="flex flex-wrap gap-x-4 gap-y-2">
            <legend className="sr-only">Lesion overlays</legend>
            {LESION_ORDER.map((code) => {
              const les = data.lesions[code];
              const ok = les?.status === 'ok';
              return (
                <label key={code} className={`flex items-center gap-2 text-sm ${ok ? '' : 'opacity-50'}`}>
                  <input
                    type="checkbox"
                    checked={ok && !!visible[code]}
                    disabled={!ok}
                    onChange={(e) => setVisible((v) => ({ ...v, [code]: e.target.checked }))}
                    className="h-4 w-4"
                  />
                  <span aria-hidden className="h-3 w-3 rounded-sm" style={{ background: rgbCss(LESION_COLORS[code]) }} />
                  {code}
                </label>
              );
            })}
          </fieldset>
          <div className="flex gap-2 text-sm">
            <button className="btn-secondary !py-1" onClick={() => setVisible(Object.fromEntries(okCodes.map((c) => [c, true])))}>
              Show all
            </button>
            <button className="btn-secondary !py-1" onClick={() => setVisible({})}>
              Hide all
            </button>
          </div>
          <label className="flex items-center gap-2 text-sm">
            Opacity
            <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Overlay opacity" />
            <span className="w-9 tabular-nums text-slate-600">{Math.round(opacity * 100)}%</span>
          </label>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <figure>
            <figcaption className="mb-1 text-sm font-medium text-slate-700">Original</figcaption>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={imageUrl} alt="Original fundus image" className="h-auto w-full rounded-lg bg-black" style={{ aspectRatio: `${W} / ${H}` }} />
          </figure>
          <figure>
            <figcaption className="mb-1 text-sm font-medium text-slate-700">
              Combined overlay {!anyVisible && <span className="font-normal text-slate-500">(no lesion layer selected)</span>}
            </figcaption>
            <canvas
              ref={overlayRef}
              role="img"
              aria-label="Fundus image with lesion segmentation overlay"
              className="h-auto w-full rounded-lg bg-black"
              style={{ aspectRatio: `${W} / ${H}` }}
            />
          </figure>
        </div>

        <ul className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-600" aria-label="Legend">
          {LESION_ORDER.map((code) => (
            <li key={code} className="flex items-center gap-1.5">
              <span aria-hidden className="h-3 w-3 rounded-sm" style={{ background: rgbCss(LESION_COLORS[code]) }} />
              {LESION_NAMES[code]} ({code})
              {data.lesions[code]?.status !== 'ok' && <em className="text-slate-400">not available</em>}
            </li>
          ))}
        </ul>
      </div>

      <div className="card space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-semibold text-slate-900">Individual mask</h2>
          <select
            aria-label="Mask to display"
            value={view}
            onChange={(e) => setView(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm"
          >
            <option value="overlay">Select a lesion…</option>
            {okCodes.map((c) => (
              <option key={c} value={c}>
                {LESION_NAMES[c]} ({c})
              </option>
            ))}
          </select>
        </div>
        <canvas
          ref={maskRef}
          role="img"
          aria-label="Binary lesion mask"
          className="h-auto w-full max-w-xl rounded-lg bg-black"
          style={{ aspectRatio: `${W} / ${H}` }}
        />
        {view === 'overlay' && <p className="text-sm text-slate-500">Choose a lesion type to view its binary mask (white = predicted lesion pixels).</p>}
      </div>

      <div className="card overflow-x-auto">
        <h2 className="mb-2 text-base font-semibold text-slate-900">Model status and results</h2>
        <table className="w-full min-w-[34rem] text-left text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="py-1 pr-3">Lesion</th>
              <th className="py-1 pr-3">Model</th>
              <th className="py-1 pr-3">Threshold</th>
              <th className="py-1 pr-3">Predicted area</th>
              <th className="py-1 pr-3">Time</th>
              <th className="py-1">Download</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {LESION_ORDER.map((code) => {
              const les: LesionResult | undefined = data.lesions[code];
              const ok = les?.status === 'ok';
              return (
                <tr key={code}>
                  <td className="py-2 pr-3">
                    <span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle" style={{ background: rgbCss(LESION_COLORS[code]) }} />
                    {LESION_NAMES[code]}
                  </td>
                  <td className="py-2 pr-3">
                    {ok ? (
                      <span className="text-emerald-700">Ready</span>
                    ) : (
                      <span className="text-red-700">{les?.status === 'error' ? 'Error' : 'Unavailable'}</span>
                    )}
                  </td>
                  <td className="py-2 pr-3 tabular-nums">{les?.threshold ?? '—'}</td>
                  <td className="py-2 pr-3 tabular-nums">{ok ? formatPercent(les.positive_fraction, 3) : '—'}</td>
                  <td className="py-2 pr-3 tabular-nums">{ok && les.inference_ms ? `${les.inference_ms.toFixed(0)} ms` : '—'}</td>
                  <td className="py-2">
                    {ok ? (
                      <button className="text-brand-700 underline" onClick={() => download(base64ToBlob(les.mask_png_base64 ?? ''), `mask_${code}.png`)}>
                        mask PNG
                      </button>
                    ) : (
                      <span className="text-xs text-red-700">{les?.error ?? 'Model not loaded'}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button className="btn-primary" onClick={saveOverlay} disabled={!original || !okCodes.length}>
            Download overlay image
          </button>
          <span className="text-xs text-slate-500">Total inference time: {data.inference_ms.toFixed(0)} ms</span>
        </div>
      </div>
    </section>
  );
}
