export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

export type Dupe = { filename: string; score: number; badge: string };

export type Keeper = {
  filename: string;
  score: number;
  badge: string;
  breakdown: Record<string, number>;
  duplicates?: Dupe[];
};

export type CullResult = {
  total: number;
  blurry_removed: number;
  duplicates_removed: number;
  unreadable_skipped: number;
  keepers: Keeper[];
  rejected: string[];
};

export type CullSettings = {
  preset: string;
  top_n: number;
  blur: number;
  dupes: number;
};

export async function cullUpload(files: File[], s: CullSettings): Promise<CullResult> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  fd.append("preset", s.preset);
  fd.append("top_n", String(s.top_n));
  fd.append("blur_threshold", String(s.blur));
  fd.append("duplicate_threshold", String(s.dupes));
  const res = await fetch(`${API_URL}/cull-upload`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`Culling failed (${res.status})`);
  return res.json();
}
