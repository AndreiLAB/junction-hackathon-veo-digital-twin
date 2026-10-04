import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, apiRaw, dash, num, pct, useApi, type Doc, type Tag } from "@/lib/api";
import { Chip, PanelChip } from "./Chip";
import { TagImage } from "./CabinetCard";
import { PdfDialog } from "./PdfDialog";
import { ErrorState } from "./States";

export function TagDrawer({ tag, onClose }: { tag: Tag | null; onClose: () => void }) {
  return (
    <Sheet open={!!tag} onOpenChange={(o) => !o && onClose()}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-xl">
        {tag && <DrawerBody initial={tag} />}
      </SheetContent>
    </Sheet>
  );
}

function DrawerBody({ initial }: { initial: Tag }) {
  const { base } = useApi();
  const qc = useQueryClient();
  const [doc, setDoc] = useState<Doc | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const q = useQuery({
    queryKey: ["tag", base, initial.id],
    queryFn: () => api<Tag>(base, `/tags/${encodeURIComponent(initial.id)}`),
    placeholderData: initial,
  });
  const tag = q.data ?? initial;

  const replace = useMutation({
    mutationFn: async (file: File) => {
      const fd = new FormData();
      fd.append("file", file);
      await apiRaw(base, `/tags/${encodeURIComponent(tag.id)}/image?placeholder=false`, {
        method: "PUT",
        body: fd,
      });
    },
    onSuccess: () => {
      toast.success("Picture replaced");
      qc.invalidateQueries();
    },
    onError: (e) => toast.error(e instanceof Error ? e.message : "Upload failed"),
  });

  const p = tag.position;
  return (
    <div className="space-y-6 pb-6">
      <SheetHeader className="text-left">
        <SheetTitle className="text-2xl">{tag.name}</SheetTitle>
        <SheetDescription>{tag.folder ?? "—"}</SheetDescription>
      </SheetHeader>
      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}
      <TagImage tag={tag} className="aspect-[4/3] rounded-lg" />
      <div>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && replace.mutate(e.target.files[0])}
        />
        <Button
          variant="outline"
          size="sm"
          onClick={() => fileRef.current?.click()}
          disabled={replace.isPending}
        >
          <Upload className="h-4 w-4" /> {replace.isPending ? "Uploading…" : "Replace picture"}
        </Button>
      </div>

      <dl className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-3">
        <Field k="Position (E57)" v={p ? `${num(p.x)}, ${num(p.y)}, ${num(p.z)}` : "—"} />
        <Field k="Confidence" v={pct(tag.confidence)} />
        <Field k="Found by E57" v={tag.found_by_e57 ? "Yes" : "No"} />
        <div className="col-span-full flex flex-wrap gap-1.5">
          <PanelChip
            model={tag.panel_model}
            source={tag.panel_source}
            confidence={tag.panel_confidence}
          />
          {tag.needs_review && <Chip variant="review">needs review</Chip>}
        </div>
      </dl>

      <section>
        {(() => {
          const uniqueDevices = Array.from(new Map(tag.devices.map((d) => [d.class, d])).values());
          return (
            <>
              <h3 className="label-mono">Devices ({uniqueDevices.length})</h3>
              {uniqueDevices.length === 0 ? (
                <p className="mt-2 text-sm text-muted-foreground">No assets detected yet</p>
              ) : (
                <ul className="mt-2 space-y-3">
                  {uniqueDevices.map((d) => (
                    <li key={d.id} className="rounded-lg border p-3 text-sm">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className="font-medium">
                          {dash(d.type)}{" "}
                          <span className="font-mono text-xs text-muted-foreground">
                            {dash(d.class)}
                          </span>
                        </span>
                        <span className="flex gap-1.5">
                          <Chip variant="detected">{pct(d.confidence)}</Chip>
                          {d.needs_review && <Chip variant="review">needs review</Chip>}
                        </span>
                      </div>
                      {d.model && <p className="mt-1 text-xs text-muted-foreground">{d.model}</p>}
                      {d.review_reasons.length > 0 && (
                        <p className="mt-1 text-xs text-destructive">
                          {d.review_reasons.join(" · ")}
                        </p>
                      )}
                      {d.documents.length > 0 && <DocList docs={d.documents} onOpen={setDoc} />}
                    </li>
                  ))}
                </ul>
              )}
            </>
          );
        })()}
      </section>

      <section>
        {(() => {
          const deviceDocs = tag.devices.flatMap((d) => d.documents);
          const allDocs = [...tag.documents, ...deviceDocs];
          const uniqueDocsMap = new Map(allDocs.map((d) => [d.id, d]));
          const uniqueDocs = Array.from(uniqueDocsMap.values());
          return (
            <>
              <h3 className="label-mono">Documents ({uniqueDocs.length})</h3>
              {uniqueDocs.length === 0 ? (
                <p className="mt-2 text-sm text-muted-foreground">No documents linked</p>
              ) : (
                <DocList docs={uniqueDocs} onOpen={setDoc} />
              )}
            </>
          );
        })()}
      </section>

      <section>
        <h3 className="label-mono">Ask the manuals</h3>
        <Input
          className="mt-2"
          disabled
          value=""
          placeholder="Available after the assets of this cabinet are detected"
          aria-label="Ask the manuals (not available yet)"
        />
      </section>
      <PdfDialog doc={doc} onClose={() => setDoc(null)} />
    </div>
  );
}

function DocList({ docs, onOpen }: { docs: Doc[]; onOpen: (d: Doc) => void }) {
  return (
    <ul className="mt-2 divide-y rounded-lg border">
      {docs.map((d) => (
        <li key={d.id} className="flex items-center justify-between gap-3 p-3 text-sm">
          <div className="min-w-0">
            <p className="flex items-center gap-1.5 font-medium">
              <FileText className="h-4 w-4 shrink-0 text-primary" />
              <span className="truncate">{d.title}</span>
            </p>
            <p className="text-xs text-muted-foreground">
              {dash(d.model)} · {d.pages ? `${d.pages} pages` : "— pages"}
            </p>
          </div>
          <Button size="sm" variant="secondary" onClick={() => onOpen(d)}>
            Open
          </Button>
        </li>
      ))}
    </ul>
  );
}

function Field({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="label-mono">{k}</dt>
      <dd className="mt-0.5 font-medium">{v}</dd>
    </div>
  );
}
