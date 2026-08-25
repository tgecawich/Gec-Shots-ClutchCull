// RAW support without RAW decoding.
//
// Every camera embeds a full JPEG preview inside its RAW file — it's what the
// camera shows on its own LCD. So we never decode sensor data (which would mean
// uploading 25-50MB per frame and running libraw on the server). We read the
// embedded JPEG in the browser, analyse that, and the RAW itself never moves.
//
// The extraction is deliberately format-agnostic: instead of parsing each
// vendor's TIFF/BMFF layout (CR2, NEF, ARW and DNG are TIFF-based; CR3 is an
// ISO-BMFF container), we scan for JPEG start/end markers and take the largest
// valid image. That single approach covers every format above, including ones
// we haven't explicitly listed.

export const RAW_EXT = /\.(cr2|cr3|nef|nrw|arw|srf|sr2|dng|raf|orf|rw2|raw|pef|srw|3fr|erf|mrw|kdc|dcr|x3f)$/i;

export function isRaw(file: File | string): boolean {
  return RAW_EXT.test(typeof file === "string" ? file : file.name);
}

/** Byte offsets of every JPEG (SOI…EOI) found in the buffer, largest first. */
function findEmbeddedJpegs(bytes: Uint8Array): Array<{ start: number; end: number }> {
  const sois: number[] = [];
  const eois: number[] = [];
  // Single pass. 0xFFD8FF starts a JPEG; 0xFFD9 ends one.
  for (let i = 0; i < bytes.length - 2; i++) {
    if (bytes[i] !== 0xff) continue;
    const b = bytes[i + 1];
    if (b === 0xd8 && bytes[i + 2] === 0xff) sois.push(i);
    else if (b === 0xd9) eois.push(i);
  }
  if (!sois.length || !eois.length) return [];

  const out: Array<{ start: number; end: number }> = [];
  let e = 0;
  for (const s of sois) {
    while (e < eois.length && eois[e] < s) e++;
    if (e >= eois.length) break;
    out.push({ start: s, end: eois[e] + 2 });
  }
  // Biggest first: the thumbnail is tiny, the preview we want is the large one.
  return out.sort((a, b) => (b.end - b.start) - (a.end - a.start));
}

/**
 * Pull the embedded JPEG preview out of a RAW file.
 * Returns null if nothing usable is found, so callers can fall back gracefully.
 */
export async function extractRawPreview(file: File): Promise<Blob | null> {
  let bytes: Uint8Array;
  try {
    bytes = new Uint8Array(await file.arrayBuffer());
  } catch {
    return null; // file unreadable (permissions, removed mid-read)
  }

  const candidates = findEmbeddedJpegs(bytes);
  // Try the largest few. Some RAWs embed a corrupt or non-standard first entry,
  // and a tiny one is just the 160px thumbnail, which is useless for judging focus.
  for (const c of candidates.slice(0, 4)) {
    const len = c.end - c.start;
    if (len < 40_000) break; // too small to be a real preview
    // Slice the underlying ArrayBuffer (bytes starts at offset 0, since it was
    // built straight from file.arrayBuffer()). Avoids a Uint8Array/BlobPart
    // typing mismatch and copies only the preview, not the whole RAW.
    const slice = bytes.buffer.slice(c.start, c.end) as ArrayBuffer;
    const blob = new Blob([slice], { type: "image/jpeg" });
    try {
      const bmp = await createImageBitmap(blob);
      const ok = bmp.width >= 800; // must be big enough to judge sharpness
      const w = bmp.width;
      bmp.close();
      if (ok) return blob;
      if (w > 0 && candidates.length === 1) return blob; // only option; take it
    } catch {
      /* not decodable — try the next candidate */
    }
  }
  return null;
}

/** Human-readable list of RAW extensions, for UI copy. */
export const RAW_ACCEPT = ".cr2,.cr3,.nef,.nrw,.arw,.sr2,.dng,.raf,.orf,.rw2,.pef,.srw";
