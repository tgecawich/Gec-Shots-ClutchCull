"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { resizeImage } from "@/lib/resize";
import { cullUpload, type CullResult, type CullSettings } from "@/lib/api";
import { makeCanvas, CANVAS_RATIOS } from "@/lib/canvas";
import { downloadZip, triggerDownload } from "@/lib/zip";
import { makeCullReport } from "@/lib/report";
import { trackSessionStart, trackPhotos, trackExport, trackCanvas, trackEmail } from "@/lib/tracking";

const PRESETS = ["Sports Action", "Portraits", "Events", "Balanced"];
const BADGE_ICON: Record<string, string> = {
  "Sharp subject": "⚡", "Clear subject": "🎯", "Rich detail": "🔍",
  "Clean contrast": "🌗", "Well-exposed": "☀️", "Strong pick": "✅",
};
const APP_LINK = "https://gec-shots-clutchcull.vercel.app";

export default function AppPage() {
  const [filesMap, setFilesMap] = useState<Record<string, File>>({});
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [results, setResults] = useState<CullResult | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [elapsed, setElapsed] = useState(0);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [dragOverC, setDragOverC] = useState(false);
  const [settings, setSettings] = useState<CullSettings>({ preset: "Balanced", top_n: 35, blur: 40, dupes: 2 });
  const [mode, setMode] = useState<"cull" | "canvas">("cull");
  const [ratio, setRatio] = useState("3:4");
  const [padding, setPadding] = useState(20);
  const [canvases, setCanvases] = useState<{ name: string; url: string; blob: Blob }[]>([]);
  const [report, setReport] = useState("");
  const [email, setEmail] = useState("");
  const [emailSaved, setEmailSaved] = useState(false);
  const [nudge, setNudge] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const canvasInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    trackSessionStart();
    if (/Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) && !localStorage.getItem("cc_nudged")) setNudge(true);
  }, []);

  const fileCount = Object.keys(filesMap).length;
  const filtered = results ? results.blurry_removed + results.duplicates_removed : 0;
  const hoursSaved = results ? (results.total * 15) / 3600 : 0;
  const canvasNames = [...selected];
  const set = (k: keyof CullSettings, v: string | number) => setSettings((s) => ({ ...s, [k]: v }));

  const ingest = useCallback((list: FileList | null, alsoSelect: boolean) => {
    if (!list) return;
    const arr = Array.from(list).filter((f) => /\.(jpe?g|png|webp)$/i.test(f.name));
    if (!arr.length) return;
    setFilesMap((prev) => { const m = { ...prev }; arr.forEach((f) => (m[f.name] = f)); return m; });
    setThumbs((prev) => { const t = { ...prev }; arr.forEach((f) => (t[f.name] ||= URL.createObjectURL(f))); return t; });
    if (alsoSelect) setSelected((prev) => { const s = new Set(prev); arr.forEach((f) => s.add(f.name)); return s; });
    else setResults(null);
  }, []);

  async function runCull() {
    if (!fileCount) return;
    setLoading(true); setError(""); setResults(null); setCanvases([]); setReport("");
    setProgress(0); setPhase("Optimizing your photos in the browser…");
    try {
      const t0 = performance.now();
      const files = Object.values(filesMap);
      let done = 0;
      const resized = await Promise.all(files.map(async (f) => {
        const r = await resizeImage(f);
        done++; setProgress((done / files.length) * 0.55);
        return r;
      }));
      setPhase("Analyzing your shoot with AI…"); setProgress(0.6);
      const res = await cullUpload(resized, settings);
      setProgress(1);
      setElapsed((performance.now() - t0) / 1000);
      setResults(res);
      setSelected(new Set(res.keepers.map((k) => k.filename)));
      trackPhotos(res.total, email);
    } catch (e: any) {
      setError(e?.message || "Something went wrong");
    } finally { setLoading(false); setPhase(""); }
  }

  const toggleSel = (name: string) =>
    setSelected((prev) => { const s = new Set(prev); s.has(name) ? s.delete(name) : s.add(name); return s; });
  const chooseFrame = (group: string[], chosen: string) =>
    setSelected((prev) => { const s = new Set(prev); group.forEach((n) => s.delete(n)); s.add(chosen); return s; });

  async function exportKeepers() {
    setBusy("Zipping full-resolution keepers…");
    const entries = [...selected].map((n, i) => ({ name: `${String(i + 1).padStart(2, "0")}_${n}`, blob: filesMap[n] as Blob })).filter((e) => e.blob);
    await downloadZip(entries, "clutchcull_keepers.zip");
    trackExport(hoursSaved * 60, selected.size, email);
    setBusy("");
  }
  const exportList = () => triggerDownload(new Blob([[...selected].join("\n") + "\n"], { type: "text/plain" }), "clutchcull_keepers.txt");
  function exportCSV() {
    if (!results) return;
    const rows = ["rank,filename,badge,score"];
    results.keepers.filter((k) => selected.has(k.filename)).forEach((k, i) => rows.push(`${i + 1},"${k.filename}","${k.badge}",${k.score.toFixed(2)}`));
    triggerDownload(new Blob([rows.join("\n") + "\n"], { type: "text/csv" }), "clutchcull_keepers.csv");
  }
  async function buildReport() {
    if (!results) return;
    setBusy("Building your shareable cull report…");
    try {
      const blob = await makeCullReport({
        shotsIn: results.total, keepers: results.keepers.length, hoursSaved,
        filtered, elapsedSeconds: elapsed,
        keeperFiles: results.keepers.slice(0, 3).map((k) => filesMap[k.filename]).filter(Boolean),
      });
      setReport(URL.createObjectURL(blob));
    } catch {}
    setBusy("");
  }
  async function generateCanvases() {
    setBusy("Building Instagram canvas posts…");
    const out: { name: string; url: string; blob: Blob }[] = [];
    for (const n of canvasNames) {
      const f = filesMap[n]; if (!f) continue;
      try { const blob = await makeCanvas(f, ratio, padding); out.push({ name: `canvas_${n.replace(/\.\w+$/, "")}.jpg`, url: URL.createObjectURL(blob), blob }); } catch {}
    }
    setCanvases(out); trackCanvas(out.length, email); setBusy("");
  }
  async function downloadCanvases() {
    setBusy("Zipping canvas posts…");
    await downloadZip(canvases.map((c) => ({ name: c.name, blob: c.blob })), "clutchcull_canvas.zip");
    setBusy("");
  }
  const saveEmail = () => { const e = email.trim(); if (e) { trackEmail(e); setEmailSaved(true); } };

  return (
    <div className="app-wrap">
      {nudge && (
        <div className="nudge">
          <span>💻 <b>ClutchCull is strongest on a computer.</b> On a phone? Open <code>{APP_LINK}</code> on a laptop for big batches.</span>
          <button onClick={() => { localStorage.setItem("cc_nudged", "1"); setNudge(false); }}>Got it</button>
        </div>
      )}
      <header className="app-top wrap">
        <a href="/" className="logo" style={{ fontSize: "1.25rem" }}>
          <span className="dot" style={{ width: 26, height: 26, fontSize: "0.85rem" }}>C</span>
          Clutch<span className="grad-text">Cull</span>
        </a>
        <a href="/" className="app-back">← Home</a>
      </header>

      <div className="wrap">
        <div className="mode-tabs">
          <button className={mode === "cull" ? "active" : ""} onClick={() => setMode("cull")}>🏟️ Cull photos</button>
          <button className={mode === "canvas" ? "active" : ""} onClick={() => setMode("canvas")}>📱 Instagram canvas</button>
        </div>
      </div>

      <main className="wrap" style={{ paddingBottom: 90 }}>
        {mode === "cull" ? (
          <>
            <h1 className="app-h1">Cull your shoot</h1>
            <p className="app-lead">Drop a full game in. AI finds your sharpest, best-framed keepers.</p>

            <div className="controls">
              <label className="control"><span>What kind of shoot?</span>
                <select value={settings.preset} onChange={(e) => set("preset", e.target.value)}>{PRESETS.map((p) => <option key={p}>{p}</option>)}</select>
              </label>
              <label className="control"><span>Keepers: <b>{settings.top_n}</b></span>
                <input type="range" min={1} max={100} value={settings.top_n} onChange={(e) => set("top_n", +e.target.value)} /></label>
              <label className="control"><span>Sharpness strictness: <b>{settings.blur}</b></span>
                <input type="range" min={0} max={100} value={settings.blur} onChange={(e) => set("blur", +e.target.value)} /></label>
              <label className="control"><span>Remove duplicates: <b>{settings.dupes}</b></span>
                <input type="range" min={0} max={10} value={settings.dupes} onChange={(e) => set("dupes", +e.target.value)} /></label>
            </div>

            <div className="trust-banner">🔒 <b>Your photos are safe.</b> Full-res originals stay on your device — nothing is sold, shared, or used to train anything.</div>

            <div className={`dropzone${dragOver ? " over" : ""}`} onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); ingest(e.dataTransfer.files, false); }}>
              <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp" multiple hidden onChange={(e) => ingest(e.target.files, false)} />
              <div className="dz-title">Drop your shoot here</div>
              <div className="dz-sub">{fileCount ? `${fileCount} photo${fileCount > 1 ? "s" : ""} ready` : "Click or drag photos — your originals stay on your device"}</div>
            </div>

            {loading && (
              <div className="cull-progress">
                <div className="cp-track">
                  {phase.startsWith("Analyzing")
                    ? <div className="cp-fill indet" />
                    : <div className="cp-fill" style={{ width: `${Math.max(6, progress * 100)}%` }} />}
                </div>
                <div className="cp-label">{phase}</div>
              </div>
            )}

            <div className="app-actions">
              <button className="btn btn-primary" disabled={!fileCount || loading} onClick={runCull}>{loading ? "Culling…" : `Cull ${fileCount || ""} photos`}</button>
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
                  <button className="btn btn-primary" disabled={!selected.size || !!busy} onClick={exportKeepers}>⬇ Download keepers (full-res ZIP)</button>
                  <button className="btn btn-ghost" disabled={!selected.size} onClick={exportList}>⬇ Download list (.txt)</button>
                  <button className="btn btn-ghost" disabled={!selected.size} onClick={exportCSV}>⬇ Download scores (.csv)</button>
                  <button className="btn btn-ghost" disabled={!!busy} onClick={buildReport}>📸 Make cull report</button>
                  <button className="btn btn-ghost" disabled={!selected.size} onClick={() => setMode("canvas")}>📱 Instagram posts →</button>
                  {busy && <span className="app-busy">{busy}</span>}
                </div>

                {report && (
                  <div className="report-out">
                    <img src={report} alt="Cull report" />
                    <a className="btn btn-primary" href={report} download="clutchcull_report.jpg">⬇ Download this card</a>
                  </div>
                )}

                <h2 className="results-h2">Your keepers <span className="muted-note">— tap to include / exclude</span></h2>
                <div className="keeper-grid">
                  {results.keepers.map((k, i) => (
                    <button className={`keeper${selected.has(k.filename) ? " sel" : ""}`} key={k.filename} onClick={() => toggleSel(k.filename)}>
                      {thumbs[k.filename] ? <img src={thumbs[k.filename]} alt={k.filename} /> : <div className="keeper-ph" />}
                      <span className="tick">{selected.has(k.filename) ? "✓" : ""}</span>
                      <div className="keeper-meta"><span className="rank">#{i + 1}</span><span className="badge">{BADGE_ICON[k.badge] || "✅"} {k.badge}</span><span className="score">{Math.round(k.score)}</span></div>
                    </button>
                  ))}
                </div>

                <details className="rank-table">
                  <summary>📊 See the scores behind these picks</summary>
                  <div className="rank-scroll"><table>
                    <thead><tr><th>#</th><th>File</th><th>Why</th><th>Score</th><th>Subject</th><th>Faces</th><th>Detail</th><th>Contrast</th><th>Exposure</th></tr></thead>
                    <tbody>{results.keepers.map((k, i) => (
                      <tr key={k.filename}><td>{i + 1}</td><td>{k.filename}</td><td>{k.badge}</td><td><b>{Math.round(k.score)}</b></td>
                        <td>{pct(k.breakdown.sharpness)}</td><td>{pct(k.breakdown.faces)}</td><td>{pct(k.breakdown.detail)}</td><td>{pct(k.breakdown.contrast)}</td><td>{pct(k.breakdown.exposure)}</td></tr>))}
                    </tbody>
                  </table></div>
                </details>

                {results.keepers.some((k) => k.duplicates && k.duplicates.length) && (
                  <>
                    <h2 className="results-h2" style={{ marginTop: 40 }}>👯 Compare similar shots <span className="muted-note">— tap another frame to swap it in</span></h2>
                    <div className="dupe-groups">
                      {results.keepers.filter((k) => k.duplicates && k.duplicates.length).map((k) => {
                        const group = [k.filename, ...k.duplicates!.map((d) => d.filename)];
                        return (<div className="dupe-group" key={k.filename}>{group.map((name) => (
                          <button className={`dupe-frame${selected.has(name) ? " sel" : ""}`} key={name} onClick={() => chooseFrame(group, name)}>
                            {thumbs[name] ? <img src={thumbs[name]} alt={name} /> : <div className="keeper-ph" />}
                            {selected.has(name) && <span className="dupe-badge">✓ Keeping</span>}
                          </button>))}</div>);
                      })}
                    </div>
                  </>
                )}

                {results.rejected.length > 0 && (
                  <>
                    <h2 className="results-h2" style={{ marginTop: 40 }}>🩹 Removed shots <span className="muted-note">— nothing is deleted; tap any to rescue</span></h2>
                    <div className="keeper-grid">{results.rejected.slice(0, 24).map((name) => (
                      <button className={`keeper removed${selected.has(name) ? " sel" : ""}`} key={name} onClick={() => toggleSel(name)}>
                        {thumbs[name] ? <img src={thumbs[name]} alt={name} /> : <div className="keeper-ph" />}
                        <span className="tick">{selected.has(name) ? "✓" : ""}</span>
                        <div className="keeper-meta"><span className="badge">{selected.has(name) ? "Rescued" : "Removed"}</span></div>
                      </button>))}
                    </div>
                  </>
                )}

                {!emailSaved ? (
                  <div className="email-capture">
                    <div><b>📊 Add your shoot to the Impact Dashboard (optional)</b><span> — email is only used for the community stats. No spam.</span></div>
                    <div className="email-row"><input type="email" placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} /><button className="btn btn-ghost" onClick={saveEmail}>Add</button></div>
                  </div>
                ) : <p className="app-busy">Thanks — you&apos;re counted in the impact stats. 🙌</p>}
              </section>
            )}
          </>
        ) : (
          <>
            <h1 className="app-h1">Instagram canvas posts</h1>
            <p className="app-lead">Drop in your picks and get clean, ready-to-post canvas versions — no culling required.</p>

            <div className={`dropzone${dragOverC ? " over" : ""}`} onClick={() => canvasInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOverC(true); }} onDragLeave={() => setDragOverC(false)}
              onDrop={(e) => { e.preventDefault(); setDragOverC(false); ingest(e.dataTransfer.files, true); }}>
              <input ref={canvasInputRef} type="file" accept=".jpg,.jpeg,.png,.webp" multiple hidden onChange={(e) => ingest(e.target.files, true)} />
              <div className="dz-title">Drop photos to make canvas posts</div>
              <div className="dz-sub">{canvasNames.length ? `${canvasNames.length} photo${canvasNames.length > 1 ? "s" : ""} ready` : "Click or drag photos — or come from Cull to use your keepers"}</div>
            </div>

            <div className="controls" style={{ gridTemplateColumns: "1fr 1fr auto" }}>
              <label className="control"><span>Post size</span>
                <select value={ratio} onChange={(e) => setRatio(e.target.value)}>{Object.keys(CANVAS_RATIOS).map((r) => <option key={r} value={r}>{r} ({CANVAS_RATIOS[r][0]}×{CANVAS_RATIOS[r][1]})</option>)}</select></label>
              <label className="control"><span>White border: <b>{padding}</b> <em style={{ color: "var(--gold)", fontStyle: "normal" }}>· Gec Shots recommends 20</em></span>
                <input type="range" min={0} max={100} value={padding} onChange={(e) => setPadding(+e.target.value)} /></label>
              <div className="control" style={{ justifyContent: "flex-end" }}>
                <button className="btn btn-primary" disabled={!canvasNames.length || !!busy} onClick={generateCanvases}>Generate {canvasNames.length || ""} posts</button></div>
            </div>

            {busy && <p className="app-busy" style={{ margin: "10px 0" }}>{busy}</p>}

            {canvases.length > 0 && (
              <>
                <div className="export-bar"><button className="btn btn-primary" onClick={downloadCanvases}>⬇ Download all {canvases.length} posts (ZIP)</button></div>
                <div className="canvas-grid">{canvases.map((c) => (<div className="canvas-item" key={c.name}><img src={c.url} alt={c.name} /></div>))}</div>
              </>
            )}
          </>
        )}
      </main>
    </div>
  );
}

const pct = (v: number | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);
