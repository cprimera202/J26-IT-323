'use client';

import { useId, useRef, useState } from 'react';

import { ACCEPT, MAX_UPLOAD_BYTES, validateImageFile, type ValidatedUpload } from '@/lib/validateImage';

interface Props {
  label: string;
  hint?: string;
  disabled?: boolean;
  onValid: (upload: ValidatedUpload) => void;
}

export default function ImageDropzone({ label, hint, disabled, onValid }: Props) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  async function handle(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setError(null);
    setChecking(true);
    try {
      onValid(await validateImageFile(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setChecking(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) void handle(e.dataTransfer.files);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors focus-within:ring-2 focus-within:ring-brand-600 ${
          disabled ? 'cursor-not-allowed opacity-50' : ''
        } ${dragging ? 'border-brand-600 bg-brand-50' : 'border-slate-300 bg-white hover:border-brand-500'}`}
      >
        <svg aria-hidden width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mb-2 text-brand-600">
          <path d="M12 16V4m0 0 4 4m-4-4-4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="text-sm font-semibold text-slate-800">{label}</span>
        <span className="mt-1 text-xs text-slate-500">
          Drag and drop, or click to browse. JPEG or PNG, up to {MAX_UPLOAD_BYTES / 1024 / 1024} MB.
        </span>
        {hint && <span className="mt-1 text-xs text-slate-500">{hint}</span>}
        <input
          id={inputId}
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          disabled={disabled}
          className="sr-only"
          onChange={(e) => void handle(e.target.files)}
        />
      </label>
      {checking && <p className="mt-2 text-sm text-slate-500" role="status">Checking image…</p>}
      {error && (
        <p role="alert" className="mt-2 rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">
          {error}
        </p>
      )}
    </div>
  );
}
