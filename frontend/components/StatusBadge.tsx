import type { ComponentStatus } from '@/lib/types';

const STYLES: Record<ComponentStatus | 'loading', { label: string; cls: string }> = {
  available: { label: 'Available', cls: 'bg-emerald-50 text-emerald-800 ring-emerald-200' },
  partial: { label: 'Partially available', cls: 'bg-amber-50 text-amber-800 ring-amber-200' },
  unavailable: { label: 'Model unavailable', cls: 'bg-red-50 text-red-800 ring-red-200' },
  under_development: { label: 'Under development', cls: 'bg-slate-100 text-slate-700 ring-slate-300' },
  service_unreachable: { label: 'Service unreachable', cls: 'bg-red-50 text-red-800 ring-red-200' },
  loading: { label: 'Checking…', cls: 'bg-slate-50 text-slate-500 ring-slate-200' },
};

export default function StatusBadge({ status }: { status: ComponentStatus | 'loading' }) {
  const s = STYLES[status];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${s.cls}`}>
      <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-current" />
      {s.label}
    </span>
  );
}
