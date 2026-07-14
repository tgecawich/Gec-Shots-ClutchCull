"use client";

import { useCallback, useRef, useState } from "react";
import { resizeImage } from "@/lib/resize";
import { cullUpload, type CullResult, type CullSettings } from "@/lib/api";

const PRESETS = ["Sports Action", "Portraits", "Events", "Balanced"];
const BADGE_ICON: Record<string, string> = {
  "Sharp subject": "⚡",
  "Clear subject": "🎯",
  "Rich detail": "🔍",
  "Clean contrast": "🌗",
  "Well-exposed": "☀️",
  "Strong pick": "✅",
};

export default function AppPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [results, setResults] = useState<CullResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [settings, setSettings] = useState<CullSettings>({ preset: "Balanced", top_n: 35, blur: 40, dupes: 2 });
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback(
    (list: FileList | null) => {
      if (!list) return;
      const arr = Array.from(list).filter((f) => /\.(jpe?g|png|webp)$/i.test(f.name));
      const byName: Record<string, File> = {};
      [...files, ...arr].forEach((f) => (byName[f.name] = f));
      setFiles(Object.values(byName));
      setThumbs((prev) => {
        const t = { ...prev };
        arr.forEach((f) => (t[f.name] ||= URL.createObjectURL(f)));
        return t;
      });
      setResults(null);
    },
    [files]
  );

  async function runCull() {
    if (!files.length) return;
    setLoading(true);
    setError("");
    try {
      const resized = await Promise.all(files.map((f) => resizeImage(f)));
      setResults(await cullUpload(resized, settings));
    } catch (e: any) {
      setError(e?.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const set = (k: keyof CullSettings, v: string | number) => setSettings((s) => ({ ...s, [k]: v }));

  return (
    <div className="app-wrap">
      <header className="app-top wrap">
        <a href="/" className="logo" style={{ fontSize: "1.25rem" }}>
          <span className="dot" style={{ width: 26, height: 26, fontSize: "0.85rem" }}>C</span>
          Clutch<span className="grad-text">Cull</span>
        </a>
        <a href="/" className="app-back">← Home</a>
      </header>

      <main className="wrap" style={{ paddingBottom: 80 }}>
        <h1 className="app-h1">Cull your shoot</h1>
        <p className="app-lead">Drop a full game in. AI finds your sharpest, best-framed keepers.</p>

        <div className="controls">
          <label className="control">
            <span>What kind of shoot?</span>
            <select value={settings.preset} onChange={(e) => set("preset", e.target.value)}>
              {PRESETS.map((p) => (<option key={p}>{p}</option>))}
            </select>
          </label>
          <label className="control">
            <span>Keepers: <b>{settings.top_n}</b></span>
            <input type="range" min={1} max={100} value={settings.top_n} onChange={(e) => set("top_n", +e.target.value)} />
          </label>
          <label className="control">
            <span>Sharpness strictness: <b>{settings.blur}</b></span>
            <input type="range" min={0} max={100} value={settings.blur} onChange={(e) => set("blur", +e.target.value)} />
          </label>
          <label className="control">
            <span>Remove duplicates: <b>{settings.dupes}</b></span>
            <input type="range" min={0} max={10} value={settings.dupes} onChange={(e) => set("dupes", +e.target.value)} />
          </label>
        </div>

        <div
          className={`dropzone${dragOver ? " over" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); addFiles(e.dataTransfer.files); }}
        >
          <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp" multiple hidden
            onChange={(e) => addFiles(e.target.files)} />
          <div className="dz-title">Drop your shoot here</div>
          <div className="dz-sub">
            {files.length ? `${files.length} photo${files.length > 1 ? "s" : ""} ready` : "Click or drag photos — your originals stay on your device"}
          </div>
        </div>

        <div className="app-actions">
          <button className="btn btn-primary" disabled={!files.length || loading} onClick={runCull}>
            {loading ? "Culling…" : `Cull ${files.length || ""} photos`}
          </button>
          {error && <span className="app-error">{error}</span>}
        </div>

        {results && (
          <section className="results">
            <div className="summary">
              <div className="chip"><b>{results.total}</b> uploaded</div>
              <div className="chip"><b>{results.blurry_removed}</b> soft removed</div>
              <div className="chip"><b>{results.duplicates_removed}</b> duplicates cut</div>
              <div className="chip chip-accent"><b>{results.keepers.length}</b> keepers</div>
            </div>

            <h2 className="results-h2">Your keepers</h2>
            <div className="keeper-grid">
              {results.keepers.map((k, i) => (
                <div className="keeper" key={k.filename}>
                  {thumbs[k.filename] ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={thumbs[k.filename]} alt={k.filename} />
                  ) : (
                    <div className="keeper-ph" />
                  )}
                  <div className="keeper-meta">
                    <span className="rank">#{i + 1}</span>
                    <span className="badge">{BADGE_ICON[k.badge] || "✅"} {k.badge}</span>
                    <span className="score">{Math.round(k.score)}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
