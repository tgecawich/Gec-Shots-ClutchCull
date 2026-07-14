"use client";

import { useCallback, useRef, useState } from "react";
import { resizeImage } from "@/lib/resize";
import { cullUpload, type CullResult, type CullSettings } from "@/lib/api";
import { makeCanvas, CANVAS_RATIOS } from "@/lib/canvas";
import { downloadZip, triggerDownload } from "@/lib/zip";

const PRESETS = ["Sports Action", "Portraits", "Events", "Balanced"];
const BADGE_ICON: Record<string, string> = {
  "Sharp subject": "⚡", "Clear subject": "🎯", "Rich detail": "🔍",
  "Clean contrast": "🌗", "Well-exposed": "☀️", "Strong pick": "✅",
};

export default function AppPage() {
  const [filesMap, setFilesMap] = useState<Record<string, File>>({});
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [results, setResults] = useState<CullResult | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [settings, setSettings] = useState<CullSettings>({ preset: "Balanced", top_n: 35, blur: 40, dupes: 2 });
  const [mode, setMode] = useState<"cull" | "canvas">("cull");
  const [ratio, setRatio] = useState("3:4");
  const [padding, setPadding] = useState(20);
  const [canvases, setCanvases] = useState<{ name: string; url: string; blob: Blob }[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const fileCount = Object.keys(filesMap).length;
  const set = (k: keyof CullSettings, v: string | number) => setSettings((s) => ({ ...s, [k]: v }));

  const addFiles = useCallback((list: FileList | null) => {
    if (!list) return;
    const arr = Array.from(list).filter((f) => /\.(jpe?g|png|webp)$/i.test(f.name));
    setFilesMap((prev) => { const m = { ...prev }; arr.forEach((f) => (m[f.name] = f)); return m; });
    setThumbs((prev) => { const t = { ...prev }; arr.forEach((f) => (t[f.name] ||= URL.createObjectURL(f))); return t; });
    setResults(null);
  }, []);

  async function runCull() {
    if (!fileCount) return;
    setLoading(true); setError(""); setResults(null); setCanvases([]);
    try {
      const files = Object.values(filesMap);
      const resized = await Promise.all(files.map((f) => resizeImage(f)));
      const res = await cullUpload(resized, settings);
      setResults(res);
      setSelected(new Set(res.keepers.map((k) => k.filename)));
    } catch (e: any) {
      setError(e?.message || "Something went wrong");
    } finally { setLoading(false); }
  }

  function toggleSel(name: string) {
    setSelected((prev) => { const s = new Set(prev); s.has(name) ? s.delete(name) : s.add(name); return s; });
  }

  async function exportKeepers() {
    setBusy("Zipping full-resolution keepers…");
    const names = [...selected];
    const entries = names.map((n, i) => ({ name: `${String(i + 1).padStart(2, "0")}_${n}`, blob: filesMap[n] as Blob }))
      .filter((e) => e.blob);
    await downloadZip(entries, "clutchcull_keepers.zip");
    setBusy("");
  }

  function exportList() {
    const txt = [...selected].join("\n") + "\n";
    triggerDownload(new Blob([txt], { type: "text/plain" }), "clutchcull_keepers.txt");
  }

  async function generateCanvases() {
    setBusy("Building Instagram canvas posts…");
    const out: { name: string; url: string; blob: Blob }[] = [];
    for (const n of selected) {
      const f = filesMap[n]; if (!f) continue;
      try {
        const blob = await makeCanvas(f, ratio, padding);
        out.push({ name: `canvas_${n.replace(/\.\w+$/, "")}.jpg`, url: URL.createObjectURL(blob), blob });
      } catch {}
    }
    setCanvases(out); setBusy("");
  }

  async function downloadCanvases() {
    setBusy("Zipping canvas posts…");
    await downloadZip(canvases.map((c) => ({ name: c.name, blob: c.blob })), "clutchcull_canvas.zip");
    setBusy("");
  }

  return (
    <div className="app-wrap">
      <header className="app-top wrap">
        <a href="/" className="logo" style={{ fontSize: "1.25rem" }}>
          <span className="dot" style={{ width: 26, height: 26, fontSize: "0.85rem" }}>C</span>
          Clutch<span className="grad-text">Cull</span>
        </a>
        <a href="/" className="app-back">← Home</a>
      </header>

      <main className="wrap" style={{ paddingBottom: 90 }}>
        {mode === "cull" ? (
          <>
            <h1 className="app-h1">Cull your shoot</h1>
            <p className="app-lead">Drop a full game in. AI finds your sharpest, best-framed keepers.</p>

            <div className="controls">
              <label className="control"><span>What kind of shoot?</span>
                <select value={settings.preset} onChange={(e) => set("preset", e.target.value)}>
                  {PRESETS.map((p) => <option key={p}>{p}</option>)}
                </select>
              </label>
              <label className="control"><span>Keepers: <b>{settings.top_n}</b></span>
                <input type="range" min={1} max={100} value={settings.top_n} onChange={(e) => set("top_n", +e.target.value)} />
              </label>
              <label className="control"><span>Sharpness strictness: <b>{settings.blur}</b></span>
                <input type="range" min={0} max={100} value={settings.blur} onChange={(e) => set("blur", +e.target.value)} />
              </label>
              <label className="control"><span>Remove duplicates: <b>{settings.dupes}</b></span>
                <input type="range" min={0} max={10} value={settings.dupes} onChange={(e) => set("dupes", +e.target.value)} />
              </label>
            </div>

            <div className={`dropzone${dragOver ? " over" : ""}`} onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); addFiles(e.dataTransfer.files); }}>
              <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp" multiple hidden
                onChange={(e) => addFiles(e.target.files)} />
              <div className="dz-title">Drop your shoot here</div>
              <div className="dz-sub">{fileCount ? `${fileCount} photo${fileCount > 1 ? "s" : ""} ready` : "Click or drag photos — your originals stay on your device"}</div>
            </div>

            <div className="app-actions">
              <button className="btn btn-primary" disabled={!fileCount || loading} onClick={runCull}>
                {loading ? "Culling…" : `Cull ${fileCount || ""} photos`}
              </button>
              {error && <span className="app-error">{error}</span>}
            </div>

            {results && (
              <section className="results">
                <div className="summary">
                  <div className="chip"><b>{results.total}</b> uploaded</div>
                  <div className="chip"><b>{results.blurry_removed}</b> soft removed</div>
                  <div className="chip"><b>{results.duplicates_removed}</b> duplicates cut</div>
                  <div className="chip chip-accent"><b>{selected.size}</b> selected</div>
                </div>

                <div className="export-bar">
                  <button className="btn btn-primary" disabled={!selected.size || !!busy} onClick={exportKeepers}>⬇ Download keepers (ZIP)</button>
                  <button className="btn btn-ghost" disabled={!selected.size} onClick={exportList}>Keeper list (.txt)</button>
                  <button className="btn btn-ghost" disabled={!selected.size} onClick={() => { setMode("canvas"); setCanvases([]); }}>📱 Create Instagram posts</button>
                  {busy && <span className="app-busy">{busy}</span>}
                </div>

                <h2 className="results-h2">Your keepers <span className="muted-note">— tap to include / exclude</span></h2>
                <div className="keeper-grid">
                  {results.keepers.map((k, i) => (
                    <button className={`keeper${selected.has(k.filename) ? " sel" : ""}`} key={k.filename} onClick={() => toggleSel(k.filename)}>
                      {thumbs[k.filename] ? <img src={thumbs[k.filename]} alt={k.filename} /> : <div className="keeper-ph" />}
                      <span className="tick">{selected.has(k.filename) ? "✓" : ""}</span>
                      <div className="keeper-meta">
                        <span className="rank">#{i + 1}</span>
                        <span className="badge">{BADGE_ICON[k.badge] || "✅"} {k.badge}</span>
                        <span className="score">{Math.round(k.score)}</span>
                      </div>
                    </button>
                  ))}
                </div>

                {results.rejected.length > 0 && (
                  <>
                    <h2 className="results-h2" style={{ marginTop: 40 }}>🩹 Removed shots <span className="muted-note">— nothing is deleted; tap any to rescue</span></h2>
                    <div className="keeper-grid">
                      {results.rejected.slice(0, 24).map((name) => (
                        <button className={`keeper removed${selected.has(name) ? " sel" : ""}`} key={name} onClick={() => toggleSel(name)}>
                          {thumbs[name] ? <img src={thumbs[name]} alt={name} /> : <div className="keeper-ph" />}
                          <span className="tick">{selected.has(name) ? "✓" : ""}</span>
                          <div className="keeper-meta"><span className="badge">{selected.has(name) ? "Rescued" : "Removed"}</span></div>
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </section>
            )}
          </>
        ) : (
          <>
            <div className="canvas-head">
              <div>
                <h1 className="app-h1">Instagram canvas posts</h1>
                <p className="app-lead">Your {selected.size} selected keeper{selected.size !== 1 ? "s" : ""}, centered on a clean white canvas.</p>
              </div>
              <button className="btn btn-ghost" onClick={() => setMode("cull")}>← Back to keepers</button>
            </div>

            <div className="controls" style={{ gridTemplateColumns: "1fr 1fr auto" }}>
              <label className="control"><span>Post size</span>
                <select value={ratio} onChange={(e) => setRatio(e.target.value)}>
                  {Object.keys(CANVAS_RATIOS).map((r) => <option key={r} value={r}>{r} ({CANVAS_RATIOS[r][0]}×{CANVAS_RATIOS[r][1]})</option>)}
                </select>
              </label>
              <label className="control"><span>White border: <b>{padding}</b> <em style={{ color: "var(--gold)", fontStyle: "normal" }}>· Gec Shots recommends 20</em></span>
                <input type="range" min={0} max={100} value={padding} onChange={(e) => setPadding(+e.target.value)} />
              </label>
              <div className="control" style={{ justifyContent: "flex-end" }}>
                <button className="btn btn-primary" disabled={!selected.size || !!busy} onClick={generateCanvases}>Generate posts</button>
              </div>
            </div>

            {busy && <p className="app-busy" style={{ margin: "10px 0" }}>{busy}</p>}

            {canvases.length > 0 && (
              <>
                <div className="export-bar"><button className="btn btn-primary" onClick={downloadCanvases}>⬇ Download all ({canvases.length}) as ZIP</button></div>
                <div className="canvas-grid">
                  {canvases.map((c) => (
                    <div className="canvas-item" key={c.name}><img src={c.url} alt={c.name} /></div>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </main>
    </div>
  );
}
