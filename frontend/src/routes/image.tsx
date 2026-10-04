import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { CameraOff, ImageOff } from "lucide-react";
import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CabinetCard } from "@/components/autotag/CabinetCard";
import { Chip } from "@/components/autotag/Chip";
import { ExportMenu } from "@/components/autotag/ExportMenu";
import { PanelCrop, PhotoOverlay } from "@/components/autotag/PhotoOverlay";
import { EmptyState, ErrorState } from "@/components/autotag/States";
import { TagDrawer } from "@/components/autotag/TagDrawer";
import { abs, api, postJson, useApi, withQuery, type ExportBody, type LocateBody, type LocateResult, type Photo, type Tag } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/image")({
  head: () => ({
    meta: [
      { title: "Image + coordinates — AutoTag360" },
      { name: "description", content: "Pick a photo or type a camera pose to see which cabinets are in view and where each asset must appear." },
      { property: "og:title", content: "Image + coordinates — AutoTag360" },
      { property: "og:description", content: "Cabinets in view from a photo or camera pose, with expected panel, relay and breaker positions." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: ImagePage,
});

type Photos = { count: number; photos: Photo[] };
const POSE_FIELDS = ["x", "y", "z", "qw", "qx", "qy", "qz"] as const;
type Pose = Record<(typeof POSE_FIELDS)[number], string>;
const emptyPose: Pose = { x: "", y: "", z: "", qw: "1", qx: "0", qy: "0", qz: "0" };

function ImagePage() {
  const { base } = useApi();
  const [category, setCategory] = useState("all");
  const [cabinet, setCabinet] = useState("all");
  const [photo, setPhoto] = useState<string | null>(null);
  const [body, setBody] = useState<LocateBody | null>(null);
  const [pose, setPose] = useState<Pose>(emptyPose);
  const [selectedCab, setSelectedCab] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<Tag | null>(null);

  const all = useQuery({ queryKey: ["photos", base, "all"], queryFn: () => api<Photos>(base, "/photos"), retry: false });
  const params = { ...(category !== "all" && { category }), ...(cabinet !== "all" && { cabinet }) };
  const filtered = useQuery({
    queryKey: ["photos", base, params],
    queryFn: () => api<Photos>(base, Object.keys(params).length ? withQuery("/photos", params) : "/photos"),
    retry: false,
  });
  const cabinetOptions = useMemo(() => [...new Set((all.data?.photos ?? []).flatMap((p) => p.cabinets))].sort(), [all.data]);
  const loc = useQuery({ queryKey: ["locate", base, body], queryFn: () => postJson<LocateResult>(base, "/locate", body), enabled: !!body, retry: false });

  const choosePhoto = (name: string) => {
    setPhoto(name);
    setSelectedCab(null);
    setBody({ photo: name });
  };
  const fillFromPhoto = () => {
    const p = all.data?.photos.find((x) => x.name === photo);
    if (!p) return;
    const [qw, qx, qy, qz] = p.rotation_wxyz.map(String) as [string, string, string, string];
    setPose({ x: String(p.position.x), y: String(p.position.y), z: String(p.position.z), qw, qx, qy, qz });
  };
  const nums = POSE_FIELDS.map((k) => Number(pose[k]));
  const poseValid = POSE_FIELDS.every((k) => pose[k].trim() !== "") && nums.every(Number.isFinite);
  const analyze = () => {
    const [x, y, z, qw, qx, qy, qz] = nums as [number, number, number, number, number, number, number];
    setPhoto(null);
    setSelectedCab(null);
    setBody({ position: { x, y, z }, rotation_wxyz: [qw, qx, qy, qz] });
  };

  const r = loc.data;
  const exportBody: ExportBody | null = body ? { method: "image", ...body } : null;
  const img1280 = r?.image_url ? withQuery(abs(base, r.image_url)!, { w: 1280 }) : null;

  return (
    <main className="mx-auto grid max-w-7xl gap-6 px-4 py-8 sm:px-6 lg:grid-cols-[340px_minmax(0,1fr)]">
      <aside className="min-w-0 space-y-4">
        <div className="flex items-center gap-2"><Chip variant="method">IMAGE</Chip><h1 className="text-xl font-semibold">Image + coordinates</h1></div>
        <Tabs defaultValue="photos">
          <TabsList className="w-full">
            <TabsTrigger value="photos" className="flex-1">Photos</TabsTrigger>
            <TabsTrigger value="coords" className="flex-1">Coordinates</TabsTrigger>
          </TabsList>
          <TabsContent value="photos" className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <Select value={category} onValueChange={setCategory}>
                <SelectTrigger aria-label="Category"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All categories</SelectItem>
                  <SelectItem value="none">None</SelectItem>
                  <SelectItem value="single">Single</SelectItem>
                  <SelectItem value="multiple">Multiple</SelectItem>
                </SelectContent>
              </Select>
              <Select value={cabinet} onValueChange={setCabinet}>
                <SelectTrigger aria-label="Cabinet"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All cabinets</SelectItem>
                  {cabinetOptions.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            {filtered.isError && <ErrorState error={filtered.error} onRetry={() => filtered.refetch()} />}
            {filtered.isLoading && <div className="grid grid-cols-2 gap-2">{Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="aspect-[4/3]" />)}</div>}
            {filtered.data && filtered.data.photos.length === 0 && <p className="text-sm text-muted-foreground">No photos match these filters.</p>}
            <div className="grid max-h-[65vh] grid-cols-2 gap-2 overflow-y-auto pr-1">
              {filtered.data?.photos.map((p) => (
                <button
                  key={p.name}
                  onClick={() => choosePhoto(p.name)}
                  className={cn("overflow-hidden rounded-lg border bg-card text-left transition hover:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring", photo === p.name && "border-primary ring-2 ring-primary/30")}
                >
                  <img src={withQuery(abs(base, p.image_url)!, { w: 320 })} alt={p.name} loading="lazy" className="aspect-[4/3] w-full bg-muted object-cover" />
                  <div className="space-y-1 p-2">
                    <p className="truncate font-mono text-[11px]">{p.name}</p>
                    <div className="flex flex-wrap gap-1">
                      <Chip variant="neutral">{p.category}</Chip>
                      {p.cabinets.length > 0 && <span className="truncate text-[11px] text-muted-foreground">{p.cabinets.join(", ")}</span>}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </TabsContent>
          <TabsContent value="coords" className="space-y-4">
            <fieldset className="space-y-2">
              <legend className="label-mono">Position (metres)</legend>
              <div className="grid grid-cols-3 gap-2">{(["x", "y", "z"] as const).map((k) => <PoseInput key={k} k={k} pose={pose} setPose={setPose} />)}</div>
            </fieldset>
            <fieldset className="space-y-2">
              <legend className="label-mono">Orientation (quaternion w, x, y, z)</legend>
              <div className="grid grid-cols-4 gap-2">{(["qw", "qx", "qy", "qz"] as const).map((k) => <PoseInput key={k} k={k} pose={pose} setPose={setPose} />)}</div>
            </fieldset>
            <div className="flex flex-wrap gap-2">
              <Button onClick={analyze} disabled={!poseValid}>Analyze</Button>
              <Button variant="outline" onClick={fillFromPhoto} disabled={!photo || !all.data}>Fill from photo</Button>
            </div>
            {!photo && <p className="text-xs text-muted-foreground">Select a photo in the Photos tab to copy its pose.</p>}
          </TabsContent>
        </Tabs>
      </aside>

      <section className="min-w-0 space-y-4">
        {!body && <EmptyState title="Pick a photo or enter a camera pose">The app shows which cabinets are in view and where each asset must appear.</EmptyState>}
        {loc.isLoading && <Skeleton className="aspect-[4/3] w-full" />}
        {loc.isError && <ErrorState error={loc.error} onRetry={() => loc.refetch()} />}
        {r && (
          <>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="font-mono text-sm">{r.photo ?? "Typed position"} · <Chip variant="neutral">{r.category}</Chip></p>
                {r.message && <p className="mt-1 text-sm text-muted-foreground">{r.message}</p>}
              </div>
              <ExportMenu body={exportBody} disabled={r.category === "none"} />
            </div>
            {r.category === "none" ? (
              <EmptyState title="No cabinet in view. No tag is created for this position." icon={<CameraOff className="h-8 w-8 text-muted-foreground" />}>
                {r.message}
              </EmptyState>
            ) : (
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
                <div className="min-w-0">
                  {img1280 ? (
                    <PhotoOverlay src={img1280} framePx={r.frame_px} cabinets={r.cabinets} selected={selectedCab} onSelect={setSelectedCab} />
                  ) : (
                    <EmptyState title="No image for a typed position" icon={<ImageOff className="h-8 w-8 text-muted-foreground" />}>The cabinets in view are listed on the right.</EmptyState>
                  )}
                  {r.note && <p className="mt-2 text-xs text-muted-foreground">{r.note}</p>}
                </div>
                <div className="space-y-3">
                  {r.cabinets.map((c) => {
                    const panel = c.assets.find((a) => a.kind === "panel");
                    return (
                      <div key={c.tag.id} onMouseEnter={() => setSelectedCab(c.tag.id)}>
                        <CabinetCard
                          tag={c.tag}
                          view={c.view}
                          selected={selectedCab === c.tag.id}
                          onOpen={() => { setSelectedCab(c.tag.id); setDrawer(c.tag); }}
                          extraPicture={img1280 && panel ? <PanelCrop src={img1280} framePx={r.frame_px} xyxy={panel.xyxy} imgRatio={1} /> : undefined}
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </section>
      <TagDrawer tag={drawer} onClose={() => setDrawer(null)} />
    </main>
  );
}

function PoseInput({ k, pose, setPose }: { k: keyof Pose; pose: Pose; setPose: (p: Pose) => void }) {
  const label = k.startsWith("q") ? k.slice(1) : k;
  return (
    <div className="space-y-1">
      <Label htmlFor={`pose-${k}`} className="text-xs">{label}</Label>
      <Input id={`pose-${k}`} inputMode="decimal" value={pose[k]} onChange={(e) => setPose({ ...pose, [k]: e.target.value })} />
    </div>
  );
}
