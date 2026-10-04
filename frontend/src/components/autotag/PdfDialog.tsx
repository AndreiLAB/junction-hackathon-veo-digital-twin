import { ExternalLink } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { abs, useApi, type Doc } from "@/lib/api";

export function PdfDialog({ doc, onClose }: { doc: Doc | null; onClose: () => void }) {
  const { base } = useApi();
  const url = doc ? abs(base, doc.url) : null;
  return (
    <Dialog open={!!doc} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="flex h-[90vh] max-w-5xl flex-col gap-3">
        <DialogHeader>
          <DialogTitle className="flex flex-wrap items-center justify-between gap-2 pr-8">
            <span className="truncate">{doc?.title}</span>
            {url && (
              <a href={url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sm font-normal text-primary underline">
                Open in new tab <ExternalLink className="h-3.5 w-3.5" />
              </a>
            )}
          </DialogTitle>
        </DialogHeader>
        {url && <iframe title={doc?.title} src={`${url}#page=1`} className="min-h-0 flex-1 rounded border" />}
      </DialogContent>
    </Dialog>
  );
}
