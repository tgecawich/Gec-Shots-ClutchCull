"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { resizeImage, mapLimit } from "@/lib/resize";
import { cullUpload, scoreUpload, rankMetrics, warmApi, type CullResult, type CullSettings, type Metric } from "@/lib/api";
import { makeCanvas, CANVAS_RATIOS } from "@/lib/canvas";
import { downloadZip, downloadZipBatched, triggerDownload } from "@/lib/zip";
import { makeCullReport } from "@/lib/report";
import { trackSessionStart, trackPhotos, trackCanvas, trackExport, trackEmail } from "@/lib/tracking";

const PRESETS = ["Sports Action", "Portraits", "Events", "Balanced"];
// Visual presets — plain labels the photographer recognises, mapped to the
// engine's internal preset names.
const PRESET_META: { value: string; label: string; icon: string; hint: string }[] = [
  { value: "Sports Action", label: "Sports", icon: "🏟️", hint: "Fast action, helmets, motion" },
  { value: "Portraits", label: "Portrait", icon: "🙂", hint: "Faces and expressions first" },
  { value: "Events", label: "Event", icon: "🎉", hint: "Mixed crowd and candids" },
  { value: "Balanced", label: "Balanced", icon: "⚖️", hint: "Good all-round default" },
];
// Measured on the live Starter instance: roughly 0.5–0.85s per photo end to end.
const SEC_PER_PHOTO_LO = 0.5;
const SEC_PER_PHOTO_HI = 0.85;
const BADGE_ICON: Record<string, string> = {
  "Sharp subject": "⚡", "Clear subject": "🎯", "Rich detail": "🔍",
  "Clean contrast": "🌗", "Well-exposed": "☀️", "Strong pick": "✅",
};
// Real deployed domain. The previous value (gec-shots-clutchcull, no hyphen)
// was dead — every mobile visitor was told to open a 404.
const APP_LINK = process.env.NEXT_PUBLIC_SITE_URL || "https://gec-shots-clutch-cull.vercel.app";

// Hand-building one padded canvas post takes ~60s (measured by Gec Shots),
// same basis as the ~15s/photo used for culling.
const CANVAS_SECONDS_EACH = 60;
// Photos rendered per view before "Show more".
const PAGE = 120;

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
  const [minutesLogged, setMinutesLogged] = useState(false);
  const [canRerank, setCanRerank] = useState(false);
  const [canvasCounted, setCanvasCounted] = useState(false);
  const [view, setView] = useState<"keepers" | "review" | "dupes" | "all">("keepers");
  const [moreOpen, setMoreOpen] = useState(false);
  const [showAdjust, setShowAdjust] = useState(false);
  const [advOpen, setAdvOpen] = useState(false);
  // Render in pages: 500 full-res originals mounted at once exhausts the
  // renderer even with lazy decoding.
  const [limit, setLimit] = useState(PAGE);
  const inputRef = useRef<HTMLInputElement>(null);
  const canvasInputRef = useRef<HTMLInputElement>(null);
  // Cached per-image metrics + the file set they belong to, so slider changes
  // re-rank instantly instead of re-uploading and re-analyzing.
  const metricsRef = useRef<{ key: string; metrics: Metric[] } | null>(null);

  useEffect(() => {
    trackSessionStart();
    warmApi(); // wake the sleeping API so the first cull isn't a cold start
    if (/Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) && !localStorage.getItem("cc_nudged")) setNudge(true);
  }, []);

  const fileCount = Object.keys(filesMap).length;
  const fileKey = Object.keys(filesMap).sort().join("|");
  const filtered = results ? results.blurry_removed + results.duplicates_removed : 0;
  const hoursSaved = results ? (results.total * 15) / 3600 : 0;
  const canvasNames = [...selected];
  const estLo = Math.max(1, Math.round((fileCount * SEC_PER_PHOTO_LO) / 60));
  const estHi = Math.max(estLo + 1, Math.round((fileCount * SEC_PER_PHOTO_HI) / 60));
  // Burst groups: keepers that had near-identical frames removed alongside them.
  const bursts = results ? results.keepers.filter((k) => k.duplicates && k.duplicates.length) : [];
  const allNames = Object.keys(filesMap);
  const set = (k: keyof CullSettings, v: string | number) => setSettings((s) => ({ ...s, [k]: v }));

  // Only the photos actually on screen right now.
  const visibleNames = useMemo(() => {
    if (!results) return [] as string[];
    if (view === "keepers") return results.keepers.slice(0, limit).map((k) => k.filename);
    if (view === "review") return results.rejected.slice(0, limit);
    if (view === "dupes") return results.keepers.filter((k) => k.duplicates?.length)
      .flatMap((k) => [k.filename, ...(k.duplicates || []).map((d) => d.filename)]);
    return Object.keys(filesMap).slice(0, limit);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results, view, limit, fileKey]);

  // Create thumbnail URLs lazily, in small batches, for visible photos only.
  // Guarded: security extensions can override URL.createObjectURL and throw when
  // it's called rapidly (they treat it as "browser locker" behaviour). A blocked
  // thumbnail must degrade to a placeholder, never take down the whole app.
  useEffect(() => {
    const missing = visibleNames.filter((n) => !thumbs[n] && filesMap[n]);
    if (!missing.length) return;
    const add: Record<string, string> = {};
    for (const n of missing.slice(0, 40)) {
      try { add[n] = URL.createObjectURL(filesMap[n]); } catch { /* blocked — placeholder */ }
    }
    if (Object.keys(add).length) setThumbs((prev) => ({ ...prev, ...add }));
  }, [visibleNames, filesMap, thumbs]);

  const ingest = useCallback((list: FileList | null, alsoSelect: boolean) => {
    if (!list) return;
    // Explain unusable files instead of silently ignoring them. Sports shooters
    // drop RAW (.CR2/.NEF/.ARW) and iPhone users drop .HEIC — previously nothing
    // happened at all, so the app looked broken and they left.
    const all = Array.from(list);
    const ok = (f: File) => /\.(jpe?g|png|webp)$/i.test(f.name);
    const arr = all.filter(ok);
    const bad = all.filter((f) => !ok(f));
    if (bad.length) {
      const exts = [...new Set(bad.map((f) => (f.name.match(/\.([^.]+)$/)?.[1] || "?").toUpperCase()))].slice(0, 4);
      const isRaw = exts.some((e) => ["CR2", "CR3", "NEF", "ARW", "RAF", "ORF", "RW2", "DNG"].includes(e));
      setError(
        arr.length
          ? `Added ${arr.length} photo${arr.length > 1 ? "s" : ""}. Skipped ${bad.length} file${bad.length > 1 ? "s" : ""} ClutchCull can't read (${exts.join(", ")}).`
          : isRaw
            ? `Those are RAW files (${exts.join(", ")}) — ClutchCull reads JPEG, PNG and WebP. Shoot RAW+JPEG on your camera, or export JPEGs from Lightroom and cull those, then apply your picks back to the RAWs.`
            : `ClutchCull reads JPEG, PNG and WebP — those are ${exts.join(", ")}. On iPhone you can set Camera → Formats → Most Compatible to shoot JPEG.`
      );
    } else {
      setError("");
    }
    if (!arr.length) return;
    warmApi(); // photos added — make sure the API is awake before they cull
    if (!alsoSelect) { metricsRef.current = null; setCanRerank(false); } // new photos → cached metrics stale
    setFilesMap((prev) => { const m = { ...prev }; arr.forEach((f) => (m[f.name] = f)); return m; });
    // Thumbnails are created lazily for on-screen photos only (see effect below).
    // Creating one per file here meant 500 synchronous URL.createObjectURL calls,
    // which some security extensions flag as "browser locker" behaviour and throw on.
    if (alsoSelect) setSelected((prev) => { const s = new Set(prev); arr.forEach((f) => s.add(f.name)); return s; });
    else setResults(null);
  }, []);

  function finishCull(res: CullResult, t0: number, fresh: boolean) {
    setProgress(1);
    setElapsed((performance.now() - t0) / 1000);
    setResults(res);
    setSelected(new Set(res.keepers.map((k) => k.filename)));
    if (fresh) trackPhotos(res.total, email); // count photos once per shoot, not per re-rank
  }

  async function runCull() {
    if (!fileCount) return;
    setLoading(true); setError(""); setCanvases([]); setReport(""); setMinutesLogged(false);
    const t0 = performance.now();
    try {
      // Fast path: we already analyzed this exact set of photos — just re-rank.
      const cached = metricsRef.current;
      if (cached && cached.key === fileKey) {
        setPhase("Ranking your keepers…"); setProgress(0.9);
        finishCull(await rankMetrics(cached.metrics, settings), t0, false);
        return;
      }

      setResults(null);
      setProgress(0); setPhase("Analyzing your shoot with AI…");
      const files = Object.values(filesMap);
      try {
        // Each chunk is resized right before it uploads (not all up front), so
        // browser memory stays flat no matter how big the shoot is.
        const metrics = await scoreUpload(
          files,
          (d, t) => setProgress((d / t) * 0.92),
          resizeImage
        );
        metricsRef.current = { key: fileKey, metrics };
        setCanRerank(true);
        setPhase("Ranking your keepers…"); setProgress(0.95);
        finishCull(await rankMetrics(metrics, settings), t0, true);
      } catch (e) {
        // Only fall back to a single-request cull for SMALL batches — a big one
        // in one request would overwhelm the server. Big batches surface the
        // error (the chunk uploader already retried each piece).
        if (files.length <= 40) {
          const resized = await mapLimit(files, 3, resizeImage);
          finishCull(await cullUpload(resized, settings), t0, true);
        } else {
          throw e;
        }
      }
    } catch (e: any) {
      const msg = e?.message || "Something went wrong";
      setError(
        fileCount > 150
          ? `${msg}. Big shoots can strain the free server — try culling in two smaller batches, or upgrade the API for full-size shoots in one go.`
          : msg
      );
    } finally {
      setLoading(false); setPhase("");
    }
  }

  // Instant re-rank: when metrics are cached and the user nudges a slider/preset,
  // rebuild keepers from the server in ~a moment — no re-upload, no re-analysis.
  useEffect(() => {
    if (!canRerank || !metricsRef.current || metricsRef.current.key !== fileKey) return;
    const id = setTimeout(async () => {
      try {
        const res = await rankMetrics(metricsRef.current!.metrics, settings);
        setResults(res);
        setSelected(new Set(res.keepers.map((k) => k.filename)));
      } catch { /* leave current results in place */ }
    }, 300);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.top_n, settings.blur, settings.dupes, settings.preset]);

  // Every download counts as an export; Hours Saved is added once per shoot.
  function logExport() {
    const mins = results && !minutesLogged ? hoursSaved * 60 : 0;
    trackExport(mins, email);
    if (results && !minutesLogged) setMinutesLogged(true);
  }

  const toggleSel = (name: string) =>
    setSelected((prev) => { const s = new Set(prev); s.has(name) ? s.delete(name) : s.add(name); return s; });
  const chooseFrame = (group: string[], chosen: string) =>
    setSelected((prev) => { const s = new Set(prev); group.forEach((n) => s.delete(n)); s.add(chosen); return s; });

  async function exportKeepers() {
    const entries = [...selected].map((n, i) => ({ name: `${String(i + 1).padStart(2, "0")}_${n}`, blob: filesMap[n] as Blob })).filter((e) => e.blob);
    if (!entries.length) return;
    setBusy("Zipping full-resolution keepers…");
    try {
      // Split big keeper sets into multiple ZIPs so a huge single blob can't
      // crash the tab. Each part downloads on its own.
      await downloadZipBatched(entries, "clutchcull_keepers", 1_200_000_000, 150, (part, total) => {
        setBusy(total > 1 ? `Zipping keepers — part ${part} of ${total}…` : "Zipping full-resolution keepers…");
      });
      logExport();
    } catch {
      setError("The keeper ZIP was too large for the browser to build at once. Try selecting fewer keepers, or download the list/scores instead.");
    } finally {
      setBusy("");
    }
  }
  function exportList() {
    triggerDownload(new Blob([[...selected].join("\n") + "\n"], { type: "text/plain" }), "clutchcull_keepers.txt");
    logExport();
  }
  function exportCSV() {
    if (!results) return;
    const rows = ["rank,filename,badge,score"];
    results.keepers.filter((k) => selected.has(k.filename)).forEach((k, i) => rows.push(`${i + 1},"${k.filename}","${k.badge}",${k.score.toFixed(2)}`));
    triggerDownload(new Blob([rows.join("\n") + "\n"], { type: "text/csv" }), "clutchcull_keepers.csv");
    logExport();
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
      try { setReport(URL.createObjectURL(blob)); } catch { setError("Couldn't build the report card — a browser extension may be blocking it."); }
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
    setCanvases(out); setCanvasCounted(false); setBusy("");
  }
  async function downloadCanvases() {
    if (!canvases.length) return;
    setBusy("Zipping canvas posts…");
    await downloadZip(canvases.map((c) => ({ name: c.name, blob: c.blob })), "clutchcull_canvas.zip");
    logExport(); // counts as an export
    // Canvas posts are photos processed AND real time saved — counted once per
    // generated batch, so re-downloading the same set doesn't inflate anything.
    if (!canvasCounted) {
      trackCanvas(canvases.length, (canvases.length * CANVAS_SECONDS_EACH) / 60, email);
      setCanvasCounted(true);
    }
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
            {!results && (
              <>
                <h1 className="app-h1">Start a new cull</h1>
                <p className="app-lead">Upload a shoot and ClutchCull will surface your strongest, sharpest frames.</p>

                {/* ---- 1. Upload (the largest, most obvious step) ---- */}
                <section className="step">
                  <div className="step-head"><span className="step-n">1</span><h2>Upload your shoot</h2></div>

                  {fileCount === 0 ? (
                    <div className={`dropzone big${dragOver ? " over" : ""}`} onClick={() => inputRef.current?.click()}
                      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
                      onDrop={(e) => { e.preventDefault(); setDragOver(false); ingest(e.dataTransfer.files, false); }}>
                      <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp" multiple hidden onChange={(e) => ingest(e.target.files, false)} />
                      <div className="dz-icon">📷</div>
                      <div className="dz-title">Drop photos or a folder here</div>
                      <div className="dz-sub">JPEG, PNG and WebP · Originals stay on your device</div>
                      <span className="btn btn-primary dz-btn">Choose photos</span>
                    </div>
                  ) : (
                    /* Once photos are in, the uploader becomes a confidence summary. */
                    <div className="ready-card">
                      <div className="ready-main">
                        <div className="ready-count">{fileCount.toLocaleString()} photos ready</div>
                        <div className="ready-meta">
                          {PRESET_META.find((p) => p.value === settings.preset)?.label} preset · Approximately {settings.top_n} keepers
                        </div>
                        <div className="ready-meta dim">Estimated processing time: {estLo}–{estHi} minutes</div>
                      </div>
                      <div className="ready-actions">
                        <button className="btn btn-primary lg" disabled={loading} onClick={runCull}>
                          {loading ? "Culling…" : `Cull ${fileCount.toLocaleString()} photos`}
                        </button>
                        <button className="linkish" disabled={loading} onClick={() => inputRef.current?.click()}>Add or replace photos</button>
                        <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp" multiple hidden onChange={(e) => ingest(e.target.files, false)} />
                      </div>
                    </div>
                  )}

                  {error && <p className="app-error">{error}</p>}

                  {loading && (
                    <div className="cull-progress">
                      <div className="cp-track"><div className="cp-fill" style={{ width: `${Math.max(4, progress * 100)}%` }} /></div>
                      <div className="cp-label">{phase} {Math.round(progress * 100)}%</div>
                    </div>
                  )}
                </section>

                {/* ---- 2. How it should review (visual presets + one goal) ---- */}
                <section className="step">
                  <div className="step-head"><span className="step-n">2</span><h2>Choose how ClutchCull should review it</h2></div>

                  <div className="preset-grid">
                    {PRESET_META.map((p) => (
                      <button key={p.value} className={`preset${settings.preset === p.value ? " on" : ""}`} onClick={() => set("preset", p.value)}>
                        <span className="preset-icon">{p.icon}</span>
                        <span className="preset-label">{p.label}</span>
                        <span className="preset-hint">{p.hint}</span>
                      </button>
                    ))}
                  </div>

                  <div className="goal">
                    <label htmlFor="keepers">How many keepers do you want?</label>
                    <div className="goal-row">
                      <span className="goal-pre">Approximately</span>
                      <input id="keepers" type="number" min={1} max={300} value={settings.top_n}
                        onChange={(e) => set("top_n", Math.max(1, Math.min(300, +e.target.value || 1)))} />
                      <span className="goal-post">photos</span>
                      <input className="goal-range" type="range" min={1} max={300} value={settings.top_n}
                        onChange={(e) => set("top_n", +e.target.value)} />
                    </div>
                  </div>

                  <button className="adv-toggle" onClick={() => setAdvOpen((v) => !v)} aria-expanded={advOpen}>
                    {advOpen ? "▾" : "▸"} Advanced preferences
                  </button>
                  {advOpen && (
                    <div className="adv-body">
                      <label className="control"><span>Sharpness strictness: <b>{settings.blur}</b></span>
                        <input type="range" min={0} max={100} value={settings.blur} onChange={(e) => set("blur", +e.target.value)} />
                        <em>Higher rejects softer frames. 40 suits most shoots.</em></label>
                      <label className="control"><span>Duplicate sensitivity: <b>{settings.dupes}</b></span>
                        <input type="range" min={0} max={10} value={settings.dupes} onChange={(e) => set("dupes", +e.target.value)} />
                        <em>Higher removes more near-identical burst frames.</em></label>
                    </div>
                  )}
                </section>

                <div className="trust-banner">🔒 <b>Your photos are safe.</b> Full-res originals stay on your device — nothing is sold, shared, or used to train anything.</div>
              </>
            )}

            {results && (
              <section className="results">
                {/* Sticky header: what happened + the one action they came for. */}
                <div className="res-bar">
                  <div className="res-sum">
                    <div className="res-title">{results.keepers.length} keepers from {results.total} photos</div>
                    <div className="res-sub">
                      {results.blurry_removed} lower-ranked · {results.duplicates_removed} duplicate{results.duplicates_removed === 1 ? "" : "s"}
                      {selected.size !== results.keepers.length && <> · <b>{selected.size} selected</b></>}
                    </div>
                  </div>
                  <div className="res-act">
                    <button className="btn btn-primary" disabled={!selected.size || !!busy} onClick={exportKeepers}>
                      Download {selected.size} keeper{selected.size === 1 ? "" : "s"}
                    </button>
                    <div className="more-wrap">
                      <button className="btn btn-ghost" onClick={() => setMoreOpen((v) => !v)} aria-expanded={moreOpen}>More export options ▾</button>
                      {moreOpen && (
                        <div className="more-menu" onMouseLeave={() => setMoreOpen(false)}>
                          <button disabled={!selected.size || !!busy} onClick={() => { setMoreOpen(false); exportKeepers(); }}>Download full-resolution ZIP</button>
                          <button disabled={!selected.size} onClick={() => { setMoreOpen(false); exportList(); }}>Export filenames (.txt)</button>
                          <button disabled={!selected.size} onClick={() => { setMoreOpen(false); exportCSV(); }}>Export scores (.csv)</button>
                          <button disabled={!!busy} onClick={() => { setMoreOpen(false); buildReport(); }}>Create cull report</button>
                          <button disabled={!selected.size} onClick={() => { setMoreOpen(false); setMode("canvas"); }}>Create Instagram canvas</button>
                        </div>
                      )}
                    </div>
                    <button className="linkish" onClick={() => setShowAdjust((v) => !v)}>Adjust results</button>
                    <button className="linkish" onClick={() => { setResults(null); setShowAdjust(false); setView("keepers"); }}>New cull</button>
                  </div>
                </div>
                {busy && <p className="app-busy">{busy}</p>}
                {error && <p className="app-error">{error}</p>}

                {showAdjust && (
                  <div className="adjust-panel">
                    <div className="preset-grid sm">
                      {PRESET_META.map((p) => (
                        <button key={p.value} className={`preset${settings.preset === p.value ? " on" : ""}`} onClick={() => set("preset", p.value)}>
                          <span className="preset-icon">{p.icon}</span><span className="preset-label">{p.label}</span>
                        </button>
                      ))}
                    </div>
                    <label className="control"><span>Keepers: <b>{settings.top_n}</b></span>
                      <input type="range" min={1} max={300} value={settings.top_n} onChange={(e) => set("top_n", +e.target.value)} /></label>
                    <label className="control"><span>Sharpness strictness: <b>{settings.blur}</b></span>
                      <input type="range" min={0} max={100} value={settings.blur} onChange={(e) => set("blur", +e.target.value)} /></label>
                    <label className="control"><span>Duplicate sensitivity: <b>{settings.dupes}</b></span>
                      <input type="range" min={0} max={10} value={settings.dupes} onChange={(e) => set("dupes", +e.target.value)} /></label>
                    <p className="adjust-note">Changes re-rank instantly — no re-upload.</p>
                  </div>
                )}

                {/* View switcher — clearer than stacked headings. */}
                <div className="views">
                  <button className={view === "keepers" ? "on" : ""} onClick={() => { setView("keepers"); setLimit(PAGE); }}>Keepers <b>{results.keepers.length}</b></button>
                  <button className={view === "review" ? "on" : ""} onClick={() => { setView("review"); setLimit(PAGE); }}>Review <b>{results.rejected.length}</b></button>
                  <button className={view === "dupes" ? "on" : ""} onClick={() => { setView("dupes"); setLimit(PAGE); }}>Bursts <b>{bursts.length}</b></button>
                  <button className={view === "all" ? "on" : ""} onClick={() => { setView("all"); setLimit(PAGE); }}>All <b>{results.total}</b></button>
                </div>
                <p className="view-hint">
                  {view === "keepers" && "ClutchCull's selections — tap any frame to include or exclude it."}
                  {view === "review" && "Lower-ranked frames. Nothing is deleted — tap any to rescue it into your keepers."}
                  {view === "dupes" && "Similar frames shot in a burst. Tap a different frame to swap which one you keep."}
                  {view === "all" && "Every photo from the shoot."}
                </p>

                {/* --- photos, immediately --- */}
                {view === "keepers" && (
                  <div className="keeper-grid">
                    {results.keepers.slice(0, limit).map((k, i) => (
                      <button className={`keeper${selected.has(k.filename) ? " sel" : ""}${k.soft ? " soft" : ""}`} key={k.filename} onClick={() => toggleSel(k.filename)}>
                        {thumbs[k.filename] ? <img src={thumbs[k.filename]} alt={k.filename} loading="lazy" decoding="async" /> : <div className="keeper-ph" />}
                        <span className="tick">{selected.has(k.filename) ? "✓" : ""}</span>
                        {k.soft && <span className="soft-flag" title="Subject looks soft — double-check before keeping">⚠ Soft</span>}
                        <div className="keeper-meta"><span className="rank">#{i + 1}</span><span className="badge">{BADGE_ICON[k.badge] || "✅"} {k.badge}</span><span className="score">{Math.round(k.score)}</span></div>
                      </button>
                    ))}
                  </div>
                )}
                {view === "keepers" && results.keepers.length > limit && (
                  <button className="show-more" onClick={() => setLimit((n) => n + PAGE)}>
                    Show more ({results.keepers.length - limit} remaining)
                  </button>
                )}

                {view === "review" && (
                  results.rejected.length ? (
                    <div className="keeper-grid">{results.rejected.slice(0, limit).map((name) => (
                      <button className={`keeper removed${selected.has(name) ? " sel" : ""}`} key={name} onClick={() => toggleSel(name)}>
                        {thumbs[name] ? <img src={thumbs[name]} alt={name} loading="lazy" decoding="async" /> : <div className="keeper-ph" />}
                        <span className="tick">{selected.has(name) ? "✓" : ""}</span>
                        <div className="keeper-meta"><span className="badge">{selected.has(name) ? "Rescued" : "Lower-ranked"}</span></div>
                      </button>))}
                    </div>
                  ) : <p className="empty-view">Nothing was filtered out of this shoot.</p>
                )}
                {view === "review" && results.rejected.length > limit && (
                  <button className="show-more" onClick={() => setLimit((n) => n + PAGE)}>
                    Show more ({results.rejected.length - limit} remaining)
                  </button>
                )}

                {view === "dupes" && (
                  bursts.length ? (
                    <div className="burst-list">
                      {bursts.slice(0, limit).map((k) => {
                        const group = [k.filename, ...k.duplicates!.map((d) => d.filename)];
                        return (
                          <div className="burst" key={k.filename}>
                            <div className="burst-head">
                              <b>Burst of {group.length} photos</b>
                              <span>ClutchCull selected {group.find((n) => selected.has(n)) === k.filename ? "this frame" : "a different frame"}</span>
                            </div>
                            <div className="dupe-group">{group.map((name) => (
                              <button className={`dupe-frame${selected.has(name) ? " sel" : ""}`} key={name} onClick={() => chooseFrame(group, name)}>
                                {thumbs[name] ? <img src={thumbs[name]} alt={name} loading="lazy" decoding="async" /> : <div className="keeper-ph" />}
                                {selected.has(name) && <span className="dupe-badge">✓ Keeping</span>}
                              </button>))}
                            </div>
                          </div>);
                      })}
                    </div>
                  ) : <p className="empty-view">No burst sequences found in this shoot.</p>
                )}

                {view === "all" && (
                  <div className="keeper-grid">{allNames.slice(0, limit).map((name) => {
                    const k = results.keepers.find((x) => x.filename === name);
                    return (
                      <button className={`keeper${selected.has(name) ? " sel" : ""}${k ? "" : " removed"}`} key={name} onClick={() => toggleSel(name)}>
                        {thumbs[name] ? <img src={thumbs[name]} alt={name} loading="lazy" decoding="async" /> : <div className="keeper-ph" />}
                        <span className="tick">{selected.has(name) ? "✓" : ""}</span>
                        <div className="keeper-meta"><span className="badge">{k ? `${BADGE_ICON[k.badge] || "✅"} ${k.badge}` : "Lower-ranked"}</span>{k && <span className="score">{Math.round(k.score)}</span>}</div>
                      </button>);
                  })}
                  </div>
                )}
                {view === "all" && allNames.length > limit && (
                  <button className="show-more" onClick={() => setLimit((n) => n + PAGE)}>
                    Show more ({allNames.length - limit} remaining)
                  </button>
                )}

                {report && (
                  <div className="report-out">
                    <img src={report} alt="Cull report" />
                    <a className="btn btn-primary" href={report} download="clutchcull_report.jpg" onClick={logExport}>⬇ Download this card</a>
                  </div>
                )}

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

                {/* Newsletter sits AFTER the work, so it never interrupts review. */}
                {!emailSaved ? (
                  <div className="email-capture">
                    <div className="ec-head">
                      <b>📬 Want new ClutchCull features first?</b>
                      <span>
                        Drop your email and I&apos;ll send you new tools as they launch — plus
                        sports-photography tips from shooting sidelines every week. Built by
                        a student-athlete photographer, free for photographers.
                      </span>
                    </div>
                    <div className="email-row">
                      <input type="email" placeholder="you@example.com" value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter") saveEmail(); }} />
                      <button className="btn btn-primary" onClick={saveEmail}>Keep me posted</button>
                    </div>
                    <div className="ec-fine">No spam, no selling your email, unsubscribe any time. Your photos are never uploaded or shared.</div>
                  </div>
                ) : (
                  <div className="email-capture done">
                    <b>🙌 You&apos;re on the list.</b>
                    <span> Thanks for supporting a student-built tool — I&apos;ll only email when there&apos;s something genuinely useful.</span>
                  </div>
                )}
              </section>
            )}
          </>
        ) : (
          <>
            <h1 className="app-h1">Make Instagram canvas posts</h1>
            <p className="app-lead">Drop in your picks and get clean, ready-to-post versions — no culling required.</p>

            <section className="step">
              <div className="step-head"><span className="step-n">1</span><h2>Add your photos</h2></div>
              {canvasNames.length === 0 ? (
                <div className={`dropzone big${dragOverC ? " over" : ""}`} onClick={() => canvasInputRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setDragOverC(true); }} onDragLeave={() => setDragOverC(false)}
                  onDrop={(e) => { e.preventDefault(); setDragOverC(false); ingest(e.dataTransfer.files, true); }}>
                  <input ref={canvasInputRef} type="file" accept=".jpg,.jpeg,.png,.webp" multiple hidden onChange={(e) => ingest(e.target.files, true)} />
                  <div className="dz-icon">📱</div>
                  <div className="dz-title">Drop photos or a folder here</div>
                  <div className="dz-sub">JPEG, PNG and WebP · Or come from Cull to use your keepers</div>
                  <span className="btn btn-primary dz-btn">Choose photos</span>
                </div>
              ) : (
                <div className="ready-card">
                  <div className="ready-main">
                    <div className="ready-count">{canvasNames.length} photo{canvasNames.length > 1 ? "s" : ""} ready</div>
                    <div className="ready-meta">{ratio} posts · {padding}px white border</div>
                  </div>
                  <div className="ready-actions">
                    <button className="btn btn-primary lg" disabled={!!busy} onClick={generateCanvases}>
                      {busy ? "Building…" : `Generate ${canvasNames.length} post${canvasNames.length > 1 ? "s" : ""}`}
                    </button>
                    <button className="linkish" onClick={() => canvasInputRef.current?.click()}>Add or replace photos</button>
                    <input ref={canvasInputRef} type="file" accept=".jpg,.jpeg,.png,.webp" multiple hidden onChange={(e) => ingest(e.target.files, true)} />
                  </div>
                </div>
              )}
              {error && <p className="app-error">{error}</p>}
            </section>

            <section className="step">
              <div className="step-head"><span className="step-n">2</span><h2>Choose your post style</h2></div>
              <div className="preset-grid">
                {Object.keys(CANVAS_RATIOS).map((r) => (
                  <button key={r} className={`preset${ratio === r ? " on" : ""}`} onClick={() => setRatio(r)}>
                    <span className="preset-label">{r}</span>
                    <span className="preset-hint">{CANVAS_RATIOS[r][0]}×{CANVAS_RATIOS[r][1]}</span>
                  </button>
                ))}
              </div>
              <div className="goal">
                <label htmlFor="pad">How much white border?</label>
                <div className="goal-row">
                  <input id="pad" type="number" min={0} max={100} value={padding}
                    onChange={(e) => setPadding(Math.max(0, Math.min(100, +e.target.value || 0)))} />
                  <span className="goal-post">px <em className="rec">· Gec Shots recommends 20</em></span>
                  <input className="goal-range" type="range" min={0} max={100} value={padding} onChange={(e) => setPadding(+e.target.value)} />
                </div>
              </div>
            </section>

            {busy && <p className="app-busy" style={{ margin: "10px 0" }}>{busy}</p>}

            {canvases.length > 0 && (
              <section className="results">
                <div className="res-bar">
                  <div className="res-sum">
                    <div className="res-title">{canvases.length} post{canvases.length > 1 ? "s" : ""} ready</div>
                    <div className="res-sub">{ratio} · {padding}px border</div>
                  </div>
                  <div className="res-act">
                    <button className="btn btn-primary" disabled={!!busy} onClick={downloadCanvases}>Download {canvases.length} post{canvases.length > 1 ? "s" : ""}</button>
                    <button className="linkish" onClick={() => setMode("cull")}>← Back to cull</button>
                  </div>
                </div>
                <div className="canvas-grid">{canvases.map((c) => (<div className="canvas-item" key={c.name}><img src={c.url} alt={c.name} loading="lazy" decoding="async" /></div>))}</div>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}

const pct = (v: number | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);
