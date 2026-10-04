import { useQuery } from "@tanstack/react-query";
import { Check, ChevronDown, Copy, Download, Info } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { apiRaw, downloadBlob, postJson, useApi, type ExportBody } from "@/lib/api";
import { ErrorState } from "./States";
import { Skeleton } from "@/components/ui/skeleton";

type Sheet = {
  count: number;
  note: string | null;
  instructions: string[];
  rows: Record<string, unknown>[];
};
type Mp = {
  count: number;
  skipped: unknown[];
  notes: string[];
  tags: unknown[];
  message?: string | null;
} & Record<string, unknown>;

export function ExportMenu({ body, disabled }: { body: ExportBody | null; disabled?: boolean }) {
  const [open, setOpen] = useState<"manual" | "mp" | null>(null);
  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button disabled={disabled || !body}>
            Export <ChevronDown className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={() => setOpen("manual")}>
            Manual tagging sheet
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => setOpen("mp")}>Matterport JSON</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {body && (
        <>
          <ManualDialog body={body} open={open === "manual"} onClose={() => setOpen(null)} />
          <MatterportDialog body={body} open={open === "mp"} onClose={() => setOpen(null)} />
        </>
      )}
    </>
  );
}

const fileTag = (b: ExportBody) =>
  b.method === "e57" ? "e57" : b.photo ? b.photo.replace(/\.\w+$/, "") : "pose";

function ManualDialog({
  body,
  open,
  onClose,
}: {
  body: ExportBody;
  open: boolean;
  onClose: () => void;
}) {
  const { base } = useApi();
  const q = useQuery({
    queryKey: ["export-manual", base, body],
    queryFn: () => postJson<Sheet>(base, "/export/manual", { ...body, format: "json" }),
    enabled: open,
  });
  const csv = async () => {
    try {
      const r = await apiRaw(base, "/export/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, format: "csv" }),
      });
      downloadBlob(await r.blob(), `manual_tagging_${fileTag(body)}.csv`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Download failed");
    }
  };
  const cols = q.data?.rows[0] ? Object.keys(q.data.rows[0]) : [];
  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-4xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Manual tagging sheet</DialogTitle>
          <DialogDescription>
            What a person creates by hand in the digital twin: name, position, picture and documents
            per cabinet.
          </DialogDescription>
        </DialogHeader>
        {q.isLoading && <Skeleton className="h-40 w-full" />}
        {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}
        {q.data &&
          (q.data.rows.length === 0 ? (
            <p className="rounded-lg bg-muted p-6 text-center text-sm text-muted-foreground">
              Nothing to export{q.data.note ? ` · ${q.data.note}` : ""}
            </p>
          ) : (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <Button onClick={csv}>
                  <Download className="h-4 w-4" /> Download CSV
                </Button>
                <Button
                  variant="outline"
                  onClick={() =>
                    downloadBlob(
                      new Blob([JSON.stringify(q.data, null, 2)], { type: "application/json" }),
                      `manual_tagging_${fileTag(body)}.json`,
                    )
                  }
                >
                  <Download className="h-4 w-4" /> Download JSON
                </Button>
              </div>
              <ul className="space-y-1.5 text-sm">
                {q.data.instructions.map((i) => (
                  <li key={i} className="flex gap-2">
                    <Check className="mt-0.5 h-4 w-4 shrink-0 text-success" />
                    {i}
                  </li>
                ))}
              </ul>
              <div className="overflow-x-auto rounded-lg border">
                <table className="w-full text-xs">
                  <thead className="bg-muted text-left">
                    <tr>
                      {cols.map((c) => (
                        <th key={c} className="whitespace-nowrap px-3 py-2 font-medium">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {q.data.rows.map((r, i) => (
                      <tr key={i} className="border-t">
                        {cols.map((c) => (
                          <td
                            key={c}
                            className="max-w-[240px] truncate px-3 py-2"
                            title={fmt(r[c])}
                          >
                            {fmt(r[c])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
      </DialogContent>
    </Dialog>
  );
}

const fmt = (v: unknown) =>
  v === null || v === undefined || v === ""
    ? "—"
    : typeof v === "object"
      ? JSON.stringify(v)
      : String(v);

function MatterportDialog({
  body,
  open,
  onClose,
}: {
  body: ExportBody;
  open: boolean;
  onClose: () => void;
}) {
  const { base } = useApi();
  const [format, setFormat] = useState<"model_api" | "sdk">("model_api");
  const [copied, setCopied] = useState(false);
  const q = useQuery({
    queryKey: ["export-mp", base, body, format],
    queryFn: () => postJson<Mp>(base, "/export/matterport", { ...body, format }),
    enabled: open,
  });
  const json = q.data ? JSON.stringify(q.data, null, 2) : "";
  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-4xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Matterport JSON</DialogTitle>
          <DialogDescription>
            <strong>Dry run: nothing is sent to Matterport.</strong> Coordinates are E57
            coordinates, not yet transformed to the Matterport model.
          </DialogDescription>
        </DialogHeader>
        <ToggleGroup
          type="single"
          value={format}
          onValueChange={(v) => v && setFormat(v as "model_api" | "sdk")}
          className="justify-start"
        >
          <ToggleGroupItem value="model_api">model_api</ToggleGroupItem>
          <ToggleGroupItem value="sdk">sdk</ToggleGroupItem>
        </ToggleGroup>
        {q.isLoading && <Skeleton className="h-60 w-full" />}
        {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}
        {q.data &&
          (q.data.tags.length === 0 ? (
            <p className="rounded-lg bg-muted p-6 text-center text-sm text-muted-foreground">
              Nothing to export{q.data.message ? ` · ${q.data.message}` : ""}
            </p>
          ) : (
            <div className="space-y-3">
              <p className="text-sm">{q.data.count} tags</p>
              {q.data.notes?.length > 0 && (
                <div className="flex gap-2 rounded-lg border border-info/30 bg-info/5 p-3 text-sm">
                  <Info className="mt-0.5 h-4 w-4 shrink-0 text-info" />
                  <ul className="space-y-1">
                    {q.data.notes.map((n) => (
                      <li key={n}>{n}</li>
                    ))}
                  </ul>
                </div>
              )}
              {q.data.skipped?.length > 0 && (
                <p className="text-sm text-warning">
                  Skipped (no position):{" "}
                  {q.data.skipped
                    .map((s) => (typeof s === "string" ? s : JSON.stringify(s)))
                    .join(", ")}
                </p>
              )}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={async () => {
                    await navigator.clipboard.writeText(json);
                    setCopied(true);
                    setTimeout(() => setCopied(false), 1500);
                  }}
                >
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />} Copy
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    downloadBlob(
                      new Blob([json], { type: "application/json" }),
                      `matterport_${format}_${fileTag(body)}.json`,
                    )
                  }
                >
                  <Download className="h-4 w-4" /> Download
                </Button>
              </div>
              <pre className="max-h-[45vh] overflow-auto rounded-lg bg-navy p-4 font-mono text-xs text-navy-foreground">
                {json}
              </pre>
            </div>
          ))}
      </DialogContent>
    </Dialog>
  );
}
