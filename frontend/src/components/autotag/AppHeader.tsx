import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Settings } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, DEFAULT_BASE, useApi } from "@/lib/api";
import { cn } from "@/lib/utils";

export function useHealth() {
  const { base } = useApi();
  return useQuery({
    queryKey: ["health", base],
    queryFn: () => api<Record<string, unknown>>(base, "/health"),
    retry: false,
    refetchInterval: 30000,
  });
}

export function AppHeader() {
  const health = useHealth();
  const [open, setOpen] = useState(false);
  const dot = health.isLoading
    ? "bg-muted-foreground"
    : health.isError
      ? "bg-destructive"
      : "bg-success";
  return (
    <header className="sticky top-0 z-40 bg-navy text-navy-foreground">
      <div className="mx-auto flex min-h-14 max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-2 sm:px-6">
        <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <span className="grid h-7 w-7 place-items-center rounded-md bg-primary text-xs font-bold text-primary-foreground">
            A
          </span>
          AutoTag360
        </Link>
        <nav aria-label="Method" className="flex rounded-lg bg-navy-foreground/10 p-1 text-sm">
          {(
            [
              ["/e57", "E57 scan"],
              ["/image", "Image + coordinates"],
            ] as const
          ).map(([to, label]) => (
            <Link
              key={to}
              to={to}
              className="rounded-md px-3 py-1.5 text-navy-foreground/75 hover:text-navy-foreground"
              activeProps={{ className: "bg-primary !text-primary-foreground" }}
            >
              {label}
            </Link>
          ))}
        </nav>
        <button
          onClick={() => setOpen(true)}
          className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-navy-foreground/10"
          aria-label="Settings"
        >
          <span className={cn("h-2.5 w-2.5 rounded-full", dot)} aria-hidden />
          <Settings className="h-5 w-5" />
        </button>
      </div>
      <SettingsDialog open={open} onOpenChange={setOpen} />
    </header>
  );
}

function SettingsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
}) {
  const { base, setBase } = useApi();
  const health = useHealth();
  const [value, setValue] = useState(base);
  useEffect(() => setValue(base), [base, open]);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Where the AutoTag360 backend runs. Saved in this browser.
          </DialogDescription>
        </DialogHeader>
        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            setBase(value.trim().replace(/\/$/, "") || DEFAULT_BASE);
          }}
        >
          <div className="space-y-1.5">
            <Label htmlFor="api-base">API base URL</Label>
            <Input
              id="api-base"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="http://localhost:8000"
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <Button type="submit">Save</Button>
            <Button type="button" variant="outline" onClick={() => setValue(DEFAULT_BASE)}>
              Reset to default
            </Button>
          </div>
        </form>
        <div className="rounded-lg border p-3 text-sm" role="status">
          {health.isLoading && <span className="text-muted-foreground">Checking connection…</span>}
          {health.isError && (
            <span className="text-destructive">
              ● Unreachable: {(health.error as Error).message}
            </span>
          )}
          {health.data && (
            <span className="text-success">
              ● Connected
              <span className="ml-2 font-mono text-xs text-muted-foreground">
                {Object.entries(health.data)
                  .filter(([, v]) => typeof v !== "object")
                  .map(([k, v]) => `${k}: ${v}`)
                  .join(" · ")}
              </span>
            </span>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function ApiBanner() {
  const health = useHealth();
  const { base } = useApi();
  if (!health.isError) return null;
  return (
    <div
      role="alert"
      className="border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-center text-sm text-destructive"
    >
      The backend at <span className="font-mono">{base}</span> is unreachable. Open Settings (gear
      icon) to change the address, or start the server.
    </div>
  );
}
