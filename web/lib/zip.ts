import JSZip from "jszip";

export async function downloadZip(
  entries: { name: string; blob: Blob }[],
  filename: string
) {
  const zip = new JSZip();
  entries.forEach((e) => zip.file(e.name, e.blob));
  const content = await zip.generateAsync({ type: "blob" });
  triggerDownload(content, filename);
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
