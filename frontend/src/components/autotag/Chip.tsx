import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

const styles = {
  assumed: "bg-warning/15 text-warning border-warning/40",
  detected: "bg-success/15 text-success border-success/40",
  review: "bg-destructive/10 text-destructive border-destructive/40",
  placeholder: "bg-muted text-muted-foreground border-border",
  expected: "bg-transparent text-info border-info",
  method: "bg-navy text-navy-foreground border-navy font-mono",
  neutral: "bg-secondary text-secondary-foreground border-border",
} as const;

export type ChipVariant = keyof typeof styles;

export function Chip({
  variant,
  children,
  className,
}: {
  variant: ChipVariant;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium",
        styles[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function PanelChip({
  model,
  source,
  confidence,
}: {
  model: string | null;
  source: "assumed" | "detected" | null;
  confidence: number | null;
}) {
  if (!model && !source) return <Chip variant="neutral">Panel —</Chip>;
  return (
    <Chip
      variant={source === "detected" ? "detected" : source === "assumed" ? "assumed" : "neutral"}
    >
      {model ?? "Panel"} · {source ?? "—"}
      {confidence != null && ` · ${Math.round(confidence * 100)}%`}
    </Chip>
  );
}
