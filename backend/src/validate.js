// Magic-byte sniffing: never trust the filename or the client-supplied MIME type.
const SIGNATURES = [
  { type: 'jpeg', mime: 'image/jpeg', ext: 'jpg', bytes: [0xff, 0xd8, 0xff] },
  { type: 'png', mime: 'image/png', ext: 'png', bytes: [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] },
];

export function sniffImage(buffer) {
  if (!buffer || buffer.length < 8) return null;
  return SIGNATURES.find((s) => s.bytes.every((b, i) => buffer[i] === b)) ?? null;
}

export const LESION_CODES = ['MA', 'HE', 'EX', 'SE'];

/** Returns a normalised comma list of lesion codes, '' for "all", or null when invalid. */
export function parseLesions(raw) {
  if (raw === undefined || raw === '') return '';
  if (typeof raw !== 'string' || !/^[A-Za-z,]{2,11}$/.test(raw)) return null;
  const parts = raw.toUpperCase().split(',').filter(Boolean);
  if (!parts.length || parts.some((p) => !LESION_CODES.includes(p))) return null;
  return [...new Set(parts)].join(',');
}
