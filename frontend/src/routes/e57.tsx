import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { AlertTriangle, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CabinetCard } from "@/components/autotag/CabinetCard";
import { Chip } from "@/components/autotag/Chip";
import { ExportMenu } from "@/components/autotag/ExportMenu";
import { CardGridSkeleton, EmptyState, ErrorState } from "@/components/autotag/States";
import { TagDrawer } from "@/components/autotag/TagDrawer";
import { api, useApi, type E57Tags, type Tag } from "@/lib/api";

export const Route = createFileRoute("/e57")({
  head: () => ({
    meta: [
      { title: "E57 scan results — AutoTag360" },
      { name: "description", content: "Every cabinet the E57 scan pipeline found, with its tag, picture and manuals." },
      { property: "og:title", content: "E57 scan results — AutoTag360" },
      { property: "og:description", content: "Cabinets found in the processed E57 scan, ready to export as tags." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: E57Page,
});

function E57Page() {
  const { base } = useApi();
  const q = useQuery({ queryKey: ["e57", base], queryFn: () => api<E57Tags>(base, "/methods/e57/tags"), retry: false });
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [open, setOpen] = useState<Tag | null>(null);
  const tags = useMemo(() => {
    const s = search.trim().toLowerCase();
    return (q.data?.tags ?? []).filter(
      (t) =>
        (!s || `${t.id} ${t.name} ${t.folder ?? ""}`.toLowerCase().includes(s)) &&
        (filter === "all" || (filter === "docs" ? t.documents.length > 0 : t.needs_review)),
    );
  }, [q.data, search, filter]);

  return (
    <main className="mx-auto max-w-7xl space-y-5 px-4 py-8 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="flex items-center gap-2"><Chip variant="method">E57</Chip><span className="label-mono">{q.data?.site ?? ""}</span></div>
          <h1 className="mt-1 text-2xl font-semibold">Cabinets found by the E57 scan {q.data && <span className="text-muted-foreground">({q.data.count})</span>}</h1>
          {q.data?.note && <p className="mt-1 text-sm text-muted-foreground">{q.data.note}</p>}
        </div>
        <ExportMenu body={{ method: "e57" }} disabled={!q.data || q.data.count === 0} />
      </div>

      {q.data && q.data.missing.length > 0 && (
        <div role="status" className="flex items-center gap-2 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm">
          <AlertTriangle className="h-4 w-4 shrink-0 text-warning" />
          Not found by the E57 scan: <strong>{q.data.missing.join(", ")}</strong>
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        <div className="relative min-w-[220px] flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search cabinets" className="pl-9" aria-label="Search cabinets" />
        </div>
        <Select value={filter} onValueChange={setFilter}>
          <SelectTrigger className="w-[200px]" aria-label="Filter"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All cabinets</SelectItem>
            <SelectItem value="docs">With documents</SelectItem>
            <SelectItem value="review">Needs review</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {q.isLoading && <CardGridSkeleton />}
      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}
      {q.data && tags.length === 0 && <EmptyState title="No cabinets match">Change the search or the filter.</EmptyState>}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {tags.map((t) => <CabinetCard key={t.id} tag={t} onOpen={() => setOpen(t)} />)}
      </div>
      <TagDrawer tag={open} onClose={() => setOpen(null)} />
    </main>
  );
}
