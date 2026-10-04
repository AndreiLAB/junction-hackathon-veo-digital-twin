// The only data seam: every value shown in the app comes from the AutoTag360 REST backend.
import { createContext, useContext } from "react";

export type Pos = { x: number; y: number; z: number };
export type Doc = {
  id: number;
  title: string;
  model: string | null;
  doc_type: string | null;
  pages: number | null;
  url: string;
};
export type Device = {
  id: number | string;
  tag_id: string | null;
  type: string | null;
  class: string | null;
  model: string | null;
  confidence: number | null;
  needs_review: boolean;
  review_reasons: string[];
  position: Pos | null;
  ocr_text: string | null;
  crop: string | null;
  documents: Doc[];
  source: { image: string | null; box: number[] | null } | null;
  xyxy?: number[] | null;
};
export type Tag = {
  id: string;
  name: string;
  folder: string | null;
  position: Pos | null;
  confidence: number | null;
  needs_review: boolean;
  image: string | null;
  image_is_placeholder: boolean;
  found_by_e57: boolean;
  panel_model: string | null;
  panel_source: "assumed" | "detected" | null;
  panel_confidence: number | null;
  documents: Doc[];
  devices: Device[];
};
export type Methods = {
  e57: { available: boolean; cabinets_found: number; cabinets_known: number; description: string };
  image: { available: boolean; photos: number; manual_pose: boolean; description: string };
};
export type E57Tags = {
  method: "e57";
  site: string;
  count: number;
  missing: string[];
  note: string | null;
  tags: Tag[];
};
export type Category = "none" | "single" | "multiple";
export type Photo = {
  name: string;
  scan: string | null;
  position: Pos;
  rotation_wxyz: number[];
  category: Category;
  cabinets: string[];
  image_url: string;
};
export type Asset = {
  class: string;
  label: string;
  kind: "panel" | "device" | "lookalike";
  xyxy: number[];
  visible_fraction: number | null;
  status: string;
};
export type LocatedCabinet = {
  tag: Tag;
  view: {
    cabinet: string;
    u: number;
    v: number;
    depth_m: number | null;
    view_angle_deg: number | null;
  };
  assets: Asset[];
  detected_devices: Device[];
};
export type LocateBody = { photo?: string; position?: Pos; rotation_wxyz?: number[] };
export type LocateResult = {
  method: "image";
  photo: string | null;
  image_url: string | null;
  pose: { position: Pos; rotation_wxyz: number[] };
  category: Category;
  message: string | null;
  frame_px: number;
  cabinets: LocatedCabinet[];
  note: string | null;
};
export type ExportBody = { method: "e57" | "image" } & LocateBody;

export const DEFAULT_BASE = (
  (import.meta.env["VITE_API_BASE_URL"] as string | undefined) ?? "http://localhost:8000"
).replace(/\/$/, "");
export const BASE_KEY = "autotag360.apiBase";

export const ApiContext = createContext<{ base: string; setBase: (b: string) => void }>({
  base: DEFAULT_BASE,
  setBase: () => {},
});
export const useApi = () => useContext(ApiContext);

/** API links are relative (`/documents/2/file`); prefix them with the base URL. */
export function abs(base: string, path: string | null | undefined) {
  if (!path) return null;
  return /^https?:\/\//.test(path) ? path : `${base}${path.startsWith("/") ? "" : "/"}${path}`;
}

export function withQuery(url: string, params: Record<string, string | number>) {
  const q = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]));
  return `${url}${url.includes("?") ? "&" : "?"}${q}`;
}

async function errorText(r: Response) {
  try {
    const j = await r.json();
    return typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
  } catch {
    return r.statusText;
  }
}

export async function apiRaw(base: string, path: string, init?: RequestInit) {
  let r: Response;
  try {
    r = await fetch(abs(base, path)!, init);
  } catch {
    throw new Error(
      `Cannot reach ${base}. Check the API base URL in Settings and that the server is running.`,
    );
  }
  if (!r.ok) throw new Error(`${r.status}: ${await errorText(r)}`);
  return r;
}

export async function api<T>(base: string, path: string, init?: RequestInit): Promise<T> {
  return (await apiRaw(base, path, init)).json() as Promise<T>;
}

export const postJson = <T>(base: string, path: string, body: unknown) =>
  api<T>(base, path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export const dash = (v: unknown) => (v === null || v === undefined || v === "" ? "—" : String(v));
export const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`;
export const num = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined ? "—" : v.toFixed(d);
