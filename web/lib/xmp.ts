// XMP sidecars, how the culling decision reaches Lightroom.
//
// You can't safely write inside a proprietary RAW, so Adobe's convention is a
// small companion file with the same basename:
//
//   D91A6535.CR3   <- your 42MB photo, untouched
//   D91A6535.xmp   <- ~1KB of XML holding the rating
//
// Lightroom, Bridge and Capture One all read it. So a photographer culls here,
// drops these next to their RAWs, hits "Read Metadata from File", and their
// keepers are already starred and filterable. The alternative is re-selecting
// every keeper by hand, which is what most people are doing today.

export type XmpEntry = {
  filename: string;   // original name, e.g. "D91A6535.CR3"
  rating: number;     // 0-5 stars
  label?: string;     // Lightroom colour label
  score?: number;     // ClutchCull score, kept for reference
  reason?: string;    // why it was picked
};

/** Strip the photo extension so the sidecar pairs with the original file. */
export function sidecarName(filename: string): string {
  return filename.replace(/\.[^.]+$/, "") + ".xmp";
}

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** A minimal, valid XMP packet carrying a star rating and colour label. */
export function makeXmp(e: XmpEntry): string {
  const desc = e.score != null
    ? `ClutchCull score ${Math.round(e.score)}${e.reason ? `, ${e.reason}` : ""}`
    : "Selected by ClutchCull";
  return `<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="ClutchCull">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmp:Rating="${e.rating}"${e.label ? `\n    xmp:Label="${esc(e.label)}"` : ""}>
   <dc:description>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">${esc(desc)}</rdf:li>
    </rdf:Alt>
   </dc:description>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>`;
}

/**
 * Build the sidecar set for a cull.
 * Keepers are starred so "filter by rating" in Lightroom shows exactly the picks.
 */
export function buildSidecars(
  keepers: Array<{ filename: string; score?: number; badge?: string }>,
  opts: { rating?: number; label?: string } = {}
): Array<{ name: string; text: string }> {
  const rating = opts.rating ?? 5;
  return keepers.map((k) => ({
    name: sidecarName(k.filename),
    text: makeXmp({ filename: k.filename, rating, label: opts.label, score: k.score, reason: k.badge }),
  }));
}
