import JSZip from "jszip";

type Entry = { name: string; blob: Blob };

// JPEGs are already compressed, so STORE (no deflate) is faster, lighter on
// memory, and just as small. Building one giant blob is what crashes the tab.
export async function downloadZip(entries: Entry[], filename: string) {
  const zip = new JSZip();
  entries.forEach((e) => zip.file(e.name, e.blob));
  const content = await zip.generateAsync({ type: "blob", compression: "STORE" });
  triggerDownload(content, filename);
}

// Split a large keeper set into several ZIPs so no single in-memory blob gets
// big enough to crash the browser (~1.2 GB cap per part). onPart reports which
// part is being built. Returns how many parts were produced.
export async function downloadZipBatched(
  entries: Entry[],
  baseName: string,
  maxBytes = 1_200_000_000,
  maxCount = 150,
  onPart?: (part: number, total: number) => void
): Promise<number> {
  const parts: Entry[][] = [];
  let cur: Entry[] = [];
  let curBytes = 0;
  for (const e of entries) {
    const size = e.blob.size || 0;
    if (cur.length && (curBytes + size > maxBytes || cur.length >= maxCount)) {
      parts.push(cur);
      cur = [];
      curBytes = 0;
    }
    cur.push(e);
    curBytes += size;
  }
  if (cur.length) parts.push(cur);

  for (let i = 0; i < parts.length; i++) {
    onPart?.(i + 1, parts.length);
    const name = parts.length === 1 ? `${baseName}.zip` : `${baseName}_part${i + 1}-of-${parts.length}.zip`;
    await downloadZip(parts[i], name);
    if (i < parts.length - 1) await new Promise((r) => setTimeout(r, 800)); // let each download start
  }
  return parts.length;
}

export function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}
