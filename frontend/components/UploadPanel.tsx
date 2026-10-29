'use client';

import type { ValidatedUpload } from '@/lib/validateImage';

/** Preview card shown after a valid image is selected. */
export default function UploadPanel({
  upload,
  children,
}: {
  upload: ValidatedUpload;
  children?: React.ReactNode;
}) {
  return (
    <div className="card">
      <div className="flex flex-col gap-4 sm:flex-row">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={upload.url}
          alt="Uploaded image preview"
          className="max-h-72 w-full rounded-lg bg-black object-contain sm:w-72"
        />
        <div className="min-w-0 flex-1 space-y-1 text-sm">
          <p className="truncate font-medium text-slate-900" title={upload.file.name}>
            {upload.file.name}
          </p>
          <p className="text-slate-500">
            {upload.width} × {upload.height} px · {(upload.file.size / 1024 / 1024).toFixed(2)} MB
          </p>
          {children}
        </div>
      </div>
    </div>
  );
}
