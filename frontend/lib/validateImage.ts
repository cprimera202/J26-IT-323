export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // keep in sync with backend MAX_UPLOAD_MB
export const MIN_SIDE = 128;
export const MAX_SIDE = 8000;
export const ACCEPT = 'image/jpeg,image/png,.jpg,.jpeg,.png';

export interface ValidatedUpload {
  file: File;
  url: string; // object URL for preview (caller must revoke)
  width: number;
  height: number;
}

export function detectFormat(bytes: Uint8Array): 'jpeg' | 'png' | null {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'jpeg';
  const png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length >= 8 && png.every((b, i) => bytes[i] === b)) return 'png';
  return null;
}

/** Cheap checks that need no decoding. Returns an error message or null. */
export function precheck(file: { size: number; name: string }): string | null {
  if (file.size === 0) return 'The selected file is empty.';
  if (file.size > MAX_UPLOAD_BYTES) return `File is too large (maximum ${MAX_UPLOAD_BYTES / 1024 / 1024} MB).`;
  return null;
}

export function dimensionError(width: number, height: number): string | null {
  if (Math.min(width, height) < MIN_SIDE) return `Image is too small (minimum ${MIN_SIDE}px on the shortest side).`;
  if (Math.max(width, height) > MAX_SIDE) return `Image is too large (maximum ${MAX_SIDE}px on the longest side).`;
  return null;
}

function loadDimensions(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error('decode'));
    img.src = url;
  });
}

/** Validates type (by file signature), size, decoding and dimensions. Throws Error(message) when invalid. */
export async function validateImageFile(file: File): Promise<ValidatedUpload> {
  const pre = precheck(file);
  if (pre) throw new Error(pre);
  const head = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  if (!detectFormat(head)) throw new Error('Unsupported file. Please upload a JPEG or PNG image.');
  const url = URL.createObjectURL(file);
  try {
    const { width, height } = await loadDimensions(url);
    const dim = dimensionError(width, height);
    if (dim) throw new Error(dim);
    return { file, url, width, height };
  } catch (err) {
    URL.revokeObjectURL(url);
    if ((err as Error).message === 'decode') throw new Error('The image could not be decoded. The file may be corrupted.');
    throw err;
  }
}
