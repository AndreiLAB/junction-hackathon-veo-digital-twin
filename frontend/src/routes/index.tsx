import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { Camera, ScanLine } from "lucide-react";
import type { ReactNode } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { Chip } from "@/components/autotag/Chip";
import { ErrorState } from "@/components/autotag/States";
import { api, useApi, type Methods } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AutoTag360 — choose a tagging method" },
      {
        name: "description",
        content:
          "Tag every switchgear cabinet automatically from the E57 scan or from a photo and its camera pose.",
      },
      { property: "og:title", content: "AutoTag360 — automatic cabinet tagging" },
      {
        property: "og:description",
        content:
          "Cabinet tags, pictures and manuals from an E57 scan or a photo, with Matterport-ready export.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Home,
});

function Home() {
  const { base } = useApi();
  const q = useQuery({
    queryKey: ["methods", base],
    queryFn: () => api<Methods>(base, "/methods"),
    retry: false,
  });
  const m = q.data;
  return (
    <main className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <p className="label-mono">Switchgear hall · digital twin</p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">Choose a tagging method</h1>
      <p className="mt-2 max-w-2xl text-muted-foreground">
        Every cabinet found gets a tag, a picture and its manuals, plus an export to create the tags
        in the digital twin.
      </p>
      {q.isError && (
        <div className="mt-6">
          <ErrorState error={q.error} onRetry={() => q.refetch()} />
        </div>
      )}
      <div className="mt-8 grid gap-6 md:grid-cols-2">
        <MethodCard
          icon={<ScanLine className="h-6 w-6" />}
          badge="E57"
          title="E57 scan"
          desc={
            m?.e57.description ??
            "Cabinets found by the E57 pipeline. The scan is already processed; no upload."
          }
          loading={q.isLoading}
          fact={m ? `${m.e57.cabinets_found} cabinets found of ${m.e57.cabinets_known} known` : "—"}
          disabledReason={
            m && !m.e57.available
              ? "The E57 pipeline has not found any cabinets on the server yet."
              : q.isError
                ? "Backend unreachable"
                : null
          }
          to="/e57"
          cta="Show cabinets"
        />
        <MethodCard
          icon={<Camera className="h-6 w-6" />}
          badge="IMAGE"
          title="Image + coordinates"
          desc={
            m?.image.description ??
            "A photo plus the pose it was taken from. No cabinet in view means no tag."
          }
          loading={q.isLoading}
          fact={
            m
              ? m.image.available
                ? `${m.image.photos} photos available`
                : "photos not available on the server"
              : "—"
          }
          disabledReason={q.isError ? "Backend unreachable" : null}
          to="/image"
          cta="Open photo viewer"
        />
      </div>
    </main>
  );
}

function MethodCard(p: {
  icon: ReactNode;
  badge: string;
  title: string;
  desc: string;
  loading: boolean;
  fact: string;
  disabledReason: string | null;
  to: "/e57" | "/image";
  cta: string;
}) {
  return (
    <div className="card-soft flex flex-col gap-4 p-6">
      <div className="flex items-center justify-between">
        <span className="grid h-11 w-11 place-items-center rounded-lg bg-accent text-primary">
          {p.icon}
        </span>
        <Chip variant="method">{p.badge}</Chip>
      </div>
      <div>
        <h2 className="text-xl font-semibold">{p.title}</h2>
        <p className="mt-1 text-sm text-muted-foreground">{p.desc}</p>
      </div>
      {p.loading ? (
        <Skeleton className="h-6 w-2/3" />
      ) : (
        <p className="text-lg font-medium text-primary">{p.fact}</p>
      )}
      <div className="mt-auto">
        {p.disabledReason ? (
          <>
            <button
              disabled
              className="w-full cursor-not-allowed rounded-md bg-muted px-4 py-2.5 text-sm font-medium text-muted-foreground"
            >
              {p.cta}
            </button>
            <p className="mt-2 text-xs text-muted-foreground">{p.disabledReason}</p>
          </>
        ) : (
          <Link
            to={p.to}
            className={cn(
              "block w-full rounded-md bg-primary px-4 py-2.5 text-center text-sm font-medium text-primary-foreground hover:bg-primary/90",
            )}
          >
            {p.cta}
          </Link>
        )}
      </div>
    </div>
  );
}
