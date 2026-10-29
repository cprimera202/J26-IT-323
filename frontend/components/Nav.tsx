'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const LINKS = [
  { href: '/', label: 'Dashboard' },
  { href: '/classification', label: 'Classification' },
  { href: '/segmentation', label: 'Segmentation' },
  { href: '/octa', label: 'OCTA' },
  { href: '/oct', label: 'OCT' },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <Link href="/" className="flex items-center gap-2 font-semibold text-brand-900">
          <span aria-hidden className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-sm text-white">
            DR
          </span>
          Diabetic Retinopathy Analysis
        </Link>
        <nav aria-label="Main" className="-mx-1 flex flex-wrap gap-1">
          {LINKS.map((l) => {
            const active = l.href === '/' ? path === '/' : path.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? 'page' : undefined}
                className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                  active ? 'bg-brand-50 text-brand-700' : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
