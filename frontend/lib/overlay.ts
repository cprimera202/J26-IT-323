/** Pure helpers for lesion overlays (kept free of DOM APIs so they are unit-testable). */

/**
 * Converts a decoded mask image (RGBA, any channel > 0 means "lesion") into a coloured RGBA buffer.
 * Alpha is 255 on lesion pixels and 0 elsewhere; global opacity is applied when drawing.
 */
export function maskToRgba(maskRgba: Uint8ClampedArray, color: [number, number, number]): Uint8ClampedArray {
  const out = new Uint8ClampedArray(maskRgba.length);
  for (let i = 0; i < maskRgba.length; i += 4) {
    if (maskRgba[i] > 127) {
      out[i] = color[0];
      out[i + 1] = color[1];
      out[i + 2] = color[2];
      out[i + 3] = 255;
    }
  }
  return out;
}

/** Same pixels, rendered as white-on-black for the standalone mask view. */
export function maskToBinaryRgba(maskRgba: Uint8ClampedArray): Uint8ClampedArray {
  const out = new Uint8ClampedArray(maskRgba.length);
  for (let i = 0; i < maskRgba.length; i += 4) {
    const v = maskRgba[i] > 127 ? 255 : 0;
    out[i] = out[i + 1] = out[i + 2] = v;
    out[i + 3] = 255;
  }
  return out;
}

export function formatPercent(fraction: number | undefined, digits = 2): string {
  if (fraction === undefined || !Number.isFinite(fraction)) return '—';
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function rgbCss(c: number[], alpha = 1): string {
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;
}

export function base64ToBlob(b64: string, type = 'image/png'): Blob {
  if (!b64) return new Blob([], { type });
  const clean = b64.includes(',') ? b64.split(',')[1] : b64;
  const bin = atob(clean.trim());
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}
