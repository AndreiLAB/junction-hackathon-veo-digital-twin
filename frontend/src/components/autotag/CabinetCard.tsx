import { FileText, ImageOff } from "lucide-react";
import type { ReactNode } from "react";
import { abs, num, useApi, type LocatedCabinet, type Tag } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Chip, PanelChip } from "./Chip";

export function TagImage({ tag, className }: { tag: Tag; className?: string }) {
  const { base } = useApi();
  const src = abs(base, tag.image);
  return (
    <div className={cn("relative overflow-hidden bg-muted", className)}>
      {src ? (
        <img src={src} alt={`Cabinet ${tag.name}`} loading="lazy" className="h-full w-full object-cover" />
      ) : (
        <div className="grid h-full w-full place-items-center text-muted-foreground">
          <ImageOff className="h-6 w-6" aria-label="No picture" />
        </div>
      )}
      {tag.image_is_placeholder && <Chip variant="placeholder" className="absolute left-2 top-2 bg-card">placeholder image</Chip>}
    </div>
  );
}

export function CabinetCard({
  tag,
  view,
  extraPicture,
  selected,
  onOpen,
}: {
  tag: Tag;
  view?: LocatedCabinet["view"];
  extraPicture?: ReactNode;
  selected?: boolean;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className={cn(
        "card-soft group flex w-full flex-col overflow-hidden text-left transition hover:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        selected && "border-primary ring-2 ring-primary/30",
      )}
    >
      {extraPicture ? (
        <div className="grid grid-cols-2 gap-px bg-border">
          <figure>
            <TagImage tag={tag} className="aspect-[4/3]" />
            <figcaption className="bg-card px-2 py-1 text-[11px] text-muted-foreground">Stored picture</figcaption>
          </figure>
          <figure>
            {extraPicture}
            <figcaption className="bg-card px-2 py-1 text-[11px] text-muted-foreground">From this photo</figcaption>
          </figure>
        </div>
      ) : (
        <TagImage tag={tag} className="aspect-[4/3]" />
      )}
      <div className="flex flex-1 flex-col gap-2 p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h3 className="truncate font-semibold">{tag.name}</h3>
            <p className="truncate text-xs text-muted-foreground">{tag.folder ?? "—"}</p>
          </div>
          {tag.needs_review && <Chip variant="review">needs review</Chip>}
        </div>
        <div className="flex flex-wrap gap-1.5">
          <PanelChip model={tag.panel_model} source={tag.panel_source} confidence={tag.panel_confidence} />
        </div>
        <div className="mt-auto flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span>{tag.devices.length ? `${tag.devices.length} device${tag.devices.length > 1 ? "s" : ""}` : "No assets detected yet"}</span>
          <span className="inline-flex items-center gap-1"><FileText className="h-3.5 w-3.5" />{tag.documents.length} documents</span>
          {view && <span>depth {num(view.depth_m, 1)} m · angle {num(view.view_angle_deg, 0)}°</span>}
        </div>
      </div>
    </button>
  );
}
