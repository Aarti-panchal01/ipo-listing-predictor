"use client";

export function Loading({ label, attempt, max }: { label: string; attempt?: number; max?: number }) {
  const waking = attempt !== undefined && attempt > 0;
  return (
    <div role="status" aria-live="polite" className="flex items-start gap-3 rounded-lg border border-line bg-surface p-4">
      <span className="mt-1 inline-block h-3 w-3 shrink-0 animate-pulse rounded-full bg-accent" aria-hidden />
      <div className="text-sm">
        <p className="font-medium">{label}</p>
        {waking && (
          <p className="mt-1 text-ink-2">
            The free prediction server sleeps when idle and is waking up (attempt {attempt + 1} of {max}). This can
            take up to a minute the first time.
          </p>
        )}
      </div>
    </div>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-lg border border-line bg-surface p-4 text-sm">
      <p className="font-medium text-bad">Something went wrong</p>
      <p className="mt-1 text-ink-2">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-md bg-ink px-3 py-1.5 text-sm font-medium text-page hover:opacity-90"
        >
          Try again
        </button>
      )}
    </div>
  );
}
