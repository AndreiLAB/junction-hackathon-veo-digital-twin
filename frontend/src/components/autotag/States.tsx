import { AlertTriangle, Inbox } from "lucide-react";
import type { ReactNode } from "react";
import { Skeleton } from "@/components/ui/skeleton";

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  return (
    <div role="alert" className="card-soft flex items-start gap-3 border-destructive/40 p-4 text-sm">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
      <div className="min-w-0">
        <p className="font-medium text-destructive">Could not load data</p>
        <p className="mt-1 break-words text-muted-foreground">{error instanceof Error ? error.message : String(error)}</p>
        {onRetry && (
          <button onClick={onRetry} className="mt-2 text-sm font-medium text-primary underline">
            Try again
          </button>
        )}
      </div>
    </div>
  );
}

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="card-soft flex flex-col items-center justify-center gap-2 p-10 text-center">
      {icon ?? <Inbox className="h-8 w-8 text-muted-foreground" />}
      <p className="text-base font-semibold">{title}</p>
      {children && <div className="max-w-md text-sm text-muted-foreground">{children}</div>}
    </div>
  );
}

export function CardGridSkeleton({ n = 6 }: { n?: number }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {Array.from({ length: n }, (_, i) => (
        <div key={i} className="card-soft overflow-hidden">
          <Skeleton className="aspect-[4/3] w-full rounded-none" />
          <div className="space-y-2 p-4">
            <Skeleton className="h-5 w-1/2" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        </div>
      ))}
    </div>
  );
}
