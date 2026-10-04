import { useState } from "react";
import type { Asset, LocatedCabinet } from "@/lib/api";
import { cn } from "@/lib/utils";

/** Colour + dash per asset class. Expected (geometry) boxes are always dashed; detected boxes are solid. */
export function assetStyle(a: Pick<Asset, "class" | "kind">) {
  const c = a.class.toLowerCase();
  if (a.kind === "panel" || c.includes("panel") || c.includes("unigear")) return { color: "var(--box-panel)", name: "Panel" };
  if (c.includes("615") || c.includes("relion")) return { color: "var(--box-relay)", name: "ABB 615 relay" };
  if (c.includes("vd4")) return { color: "var(--box-vd4)", name: "VD4 window" };
  return { color: "var(--box-other)", name: "Look-alike display" };
}

export function PhotoOverlay({
  src,
  framePx,
  cabinets,
  selected,
  onSelect,
}: {
  src: string;
  framePx: number;
  cabinets: LocatedCabinet[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const [ratio, setRatio] = useState<number | null>(null);
  const h = ratio ? framePx / ratio : framePx * 0.75;
  const font = framePx / 70;
  return (
    <div className="space-y-2">
      <div className="relative overflow-hidden rounded-lg bg-muted">
        <img src={src} alt="Selected photo" className="block w-full" onLoad={(e) => setRatio(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)} />
        {ratio && (
          <svg viewBox={`0 0 ${framePx} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
            {cabinets.map((cab) => {
              const sel = selected === cab.tag.id;
              return (
                <g key={cab.tag.id} className="cursor-pointer" onClick={() => onSelect(cab.tag.id)} opacity={selected && !sel ? 0.45 : 1}>
                  {cab.assets.map((a, i) => {
                    const s = assetStyle(a);
                    const [x0, y0, x1, y1] = a.xyxy as [number, number, number, number];
                    return (
                      <g key={i}>
                        <title>{`${cab.tag.name} · ${a.label} (${a.class}) · expected position (geometry)`}</title>
                        <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="transparent" stroke={s.color} strokeWidth={sel ? 3 : 2} strokeDasharray="8 6" vectorEffect="non-scaling-stroke" />
                        {a.kind === "panel" && (
                          <text x={x0 + 8} y={y0 + font * 1.1} fontSize={font} fill={s.color} fontWeight={600} style={{ paintOrder: "stroke", stroke: "black", strokeWidth: font / 6 }}>
                            {cab.tag.name} · expected position (geometry)
                          </text>
                        )}
                      </g>
                    );
                  })}
                  {cab.detected_devices.filter((d) => d.xyxy).map((d) => {
                    const [x0, y0, x1, y1] = d.xyxy as [number, number, number, number];
                    const s = assetStyle({ class: d.class ?? "", kind: "device" });
                    return (
                      <g key={d.id}>
                        <title>{`${d.type ?? "device"} (${d.class ?? "—"}) · detected ${d.confidence != null ? Math.round(d.confidence * 100) + "%" : ""}`}</title>
                        <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="transparent" stroke={s.color} strokeWidth={3} vectorEffect="non-scaling-stroke" />
                        <text x={x0} y={y0 - 6} fontSize={font} fill={s.color} fontWeight={700} style={{ paintOrder: "stroke", stroke: "black", strokeWidth: font / 6 }}>
                          detected {d.confidence != null ? `${Math.round(d.confidence * 100)}%` : ""}
                        </text>
                      </g>
                    );
                  })}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground" aria-label="Legend">
        {[
          ["var(--box-panel)", "Panel"],
          ["var(--box-relay)", "ABB 615 relay"],
          ["var(--box-vd4)", "VD4 window"],
          ["var(--box-other)", "Look-alike display"],
        ].map(([c, n]) => (
          <li key={n} className="flex items-center gap-1.5"><span className="h-3 w-4 rounded-sm border-2" style={{ borderColor: c }} />{n}</li>
        ))}
        <li className="flex items-center gap-1.5"><span className="h-3 w-4 border-2 border-dashed border-foreground/60" />expected (geometry)</li>
        <li className="flex items-center gap-1.5"><span className="h-3 w-4 border-2 border-foreground/60" />detected</li>
      </ul>
    </div>
  );
}

/** Crop of the 1280-px photo around the panel box (boxes are in frame_px coordinates). */
export function PanelCrop({ src, framePx, xyxy, imgRatio, className }: { src: string; framePx: number; xyxy: number[]; imgRatio: number; className?: string }) {
  const [x0, y0, x1, y1] = xyxy as [number, number, number, number];
  const w = Math.max(1, x1 - x0);
  const hBox = Math.max(1, y1 - y0);
  void imgRatio;
  return (
    <div className={cn("relative aspect-[4/3] overflow-hidden bg-muted", className)}>
      <div className="absolute inset-0 grid place-items-center">
        <div className="relative w-full overflow-hidden" style={{ aspectRatio: `${w} / ${hBox}`, maxHeight: "100%" }}>
          <img src={src} alt="Cabinet from this photo" className="absolute max-w-none" style={{ width: `${(framePx / w) * 100}%`, left: `${(-x0 / w) * 100}%`, top: `${(-y0 / hBox) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
