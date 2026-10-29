import { describe, expect, it } from 'vitest';

import { formatPercent, maskToBinaryRgba, maskToRgba } from '@/lib/overlay';
import { LESION_COLORS } from '@/lib/types';
import { dimensionError, detectFormat, precheck } from '@/lib/validateImage';

describe('overlay helpers', () => {
  const mask = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255]);

  it('colours only lesion pixels and keeps pixel positions', () => {
    const out = maskToRgba(mask, [255, 0, 0]);
    expect([...out.slice(0, 4)]).toEqual([255, 0, 0, 255]);
    expect([...out.slice(4, 8)]).toEqual([0, 0, 0, 0]);
    expect([...out.slice(8, 12)]).toEqual([255, 0, 0, 255]);
  });

  it('produces an opaque binary mask view', () => {
    const out = maskToBinaryRgba(mask);
    expect([...out.slice(0, 4)]).toEqual([255, 255, 255, 255]);
    expect([...out.slice(4, 8)]).toEqual([0, 0, 0, 255]);
  });

  it('uses the required lesion colours', () => {
    expect(LESION_COLORS.MA).toEqual([255, 0, 0]);
    expect(LESION_COLORS.HE).toEqual([0, 128, 255]);
    expect(LESION_COLORS.EX[0]).toBe(255); // yellow
    expect(LESION_COLORS.SE[1]).toBeGreaterThan(150); // green
  });

  it('formats percentages safely', () => {
    expect(formatPercent(0.01234, 2)).toBe('1.23%');
    expect(formatPercent(undefined)).toBe('—');
    expect(formatPercent(NaN)).toBe('—');
  });
});

describe('upload validation', () => {
  it('detects JPEG and PNG by signature, not by name', () => {
    expect(detectFormat(new Uint8Array([0xff, 0xd8, 0xff, 0xe0]))).toBe('jpeg');
    expect(detectFormat(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))).toBe('png');
    expect(detectFormat(new TextEncoder().encode('<html>hello</html>'))).toBeNull();
    expect(detectFormat(new TextEncoder().encode('GIF89a......'))).toBeNull();
  });

  it('rejects empty and oversized files', () => {
    expect(precheck({ size: 0, name: 'a.png' })).toMatch(/empty/);
    expect(precheck({ size: 11 * 1024 * 1024, name: 'a.png' })).toMatch(/too large/);
    expect(precheck({ size: 1024, name: 'a.png' })).toBeNull();
  });

  it('rejects too-small and too-large dimensions', () => {
    expect(dimensionError(64, 64)).toMatch(/too small/);
    expect(dimensionError(9000, 500 + 200)).toMatch(/too large/);
    expect(dimensionError(1024, 768)).toBeNull();
  });
});
