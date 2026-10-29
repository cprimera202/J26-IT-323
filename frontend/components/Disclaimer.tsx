export const DISCLAIMER_TEXT =
  'Research use only. This tool is not a medical device and its output must not be used for standalone clinical diagnosis.';

export default function Disclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <p
      role="note"
      className={`rounded-lg border border-amber-200 bg-amber-50 text-amber-900 ${compact ? 'px-3 py-2 text-xs' : 'px-4 py-3 text-sm'}`}
    >
      <strong className="font-semibold">Research use only.</strong> Not a medical device. Results are model outputs, not a confirmed
      diagnosis, and must not be used for standalone clinical decisions.
    </p>
  );
}
