const APP_URL = process.env.NEXT_PUBLIC_APP_URL || "/app";

const SHEET_CSV =
  "https://docs.google.com/spreadsheets/d/e/2PACX-1vQi3K-ggYir5zDj6mYXIMrWcG-fyTqWu6tQtQ2g97vFpzSZmKrJ0nnExHIBzA7zDCbvhabDdCG8EYSa/pub?gid=370944124&single=true&output=csv";

// Live Impact Dashboard — fetched server-side from the shared Google Sheet
// (same data source as the original app), so it stays visible and current.
async function getImpact() {
  try {
    const res = await fetch(SHEET_CSV, { next: { revalidate: 300 } });
    if (!res.ok) return null;
    const rows = (await res.text()).trim().split(/\r?\n/).map((r) => r.split(","));
    const header = rows[0].map((h) => h.trim().toLowerCase().replace(/ /g, "_"));
    const ci = (n: string) => header.indexOf(n);
    const iE = ci("event_type"), iP = ci("photos_processed"), iX = ci("exports"), iM = ci("minutes_saved");
    let sessions = 0, photos = 0, exports = 0, minutes = 0;
    for (const row of rows.slice(1)) {
      if (iE >= 0 && row[iE] === "session_start") sessions++;
      if (iP >= 0) photos += parseFloat(row[iP]) || 0;
      if (iX >= 0) exports += parseFloat(row[iX]) || 0;
      if (iM >= 0) minutes += parseFloat(row[iM]) || 0;
    }
    return { sessions, photos, exports, hours: minutes / 60 };
  } catch {
    return null;
  }
}

const shots = [
  { tag: "⚡ Sharp subject", score: 98 },
  { tag: "🎯 Clear subject", score: 95 },
  { tag: "☀ Well-exposed", score: 91 },
  { tag: "⚡ Sharp subject", score: 88 },
  { tag: "🔍 Rich detail", score: 86 },
  { tag: "🌗 Clean contrast", score: 83 },
];

const features = [
  {
    title: "AI subject detection",
    body: "Locks onto the player so a sharp background behind a blurry athlete never wins.",
    icon: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7z" />
      </>
    ),
  },
  {
    title: "Nothing is deleted",
    body: "Removed a keeper? Rescue any shot from the bin in one tap. Your call is always final.",
    icon: (
      <>
        <path d="M3 12a9 9 0 1 0 9-9" />
        <path d="M3 4v5h5" />
      </>
    ),
  },
  {
    title: "Instagram canvas",
    body: "Turn picks into ready-to-post 3:4, 4:5, or 1:1 canvas posts in a tap.",
    icon: (
      <>
        <rect x="4" y="3" width="16" height="18" rx="2" />
        <path d="M4 15l4-4 4 4 4-5 4 4" />
      </>
    ),
  },
  {
    title: "Your photos stay private",
    body: "Full-res originals never leave your device. Nothing is sold, shared, or used to train anything.",
    icon: (
      <>
        <rect x="5" y="11" width="14" height="10" rx="2" />
        <path d="M8 11V7a4 4 0 0 1 8 0v4" />
      </>
    ),
  },
  {
    title: "Blazing fast uploads",
    body: "Photos optimize in your browser and upload in parallel — up to 50× faster on big batches.",
    icon: <path d="M13 2 3 14h7l-1 8 10-12h-7z" />,
  },
  {
    title: "Transparent ranking",
    body: "Every keeper shows why it was picked — sharpness, subject, exposure — no black box.",
    icon: (
      <>
        <path d="M3 3v18h18" />
        <path d="M7 14l4-4 3 3 5-6" />
      </>
    ),
  },
];

const steps = [
  { n: "01", title: "Drop your shoot", body: "Upload a whole game — hundreds of frames, straight from your browser." },
  { n: "02", title: "AI ranks the keepers", body: "Blurry frames and duplicates out; your sharpest, best-framed shots surfaced and scored." },
  { n: "03", title: "Export & post", body: "Grab full-res keepers or one-tap Instagram canvas posts. You're done." },
];

const stats = [
  { num: "428→35", lbl: "shots to keepers" },
  { num: "0:19", lbl: "avg cull time" },
  { num: "15,131", lbl: "photos processed" },
  { num: "30.6", lbl: "hours saved" },
];

export default async function Home() {
  const impact = await getImpact();
  const dash = impact
    ? [
        { num: impact.sessions.toLocaleString(), lbl: "Sessions" },
        { num: Math.round(impact.photos).toLocaleString(), lbl: "Photos processed" },
        { num: Math.round(impact.exports).toLocaleString(), lbl: "Exports" },
        { num: impact.hours.toFixed(1), lbl: "Hours saved" },
      ]
    : stats;
  return (
    <>
      <nav>
        <div className="wrap nav-in">
          <div className="logo">
            <span className="dot">C</span>Clutch<span className="grad-text">Cull</span>
          </div>
          <div className="nav-links">
            <a href="#features">Features</a>
            <a href="#how">How it works</a>
            <a href="#">Canvas</a>
          </div>
          <a className="btn btn-primary" href={APP_URL}>Launch app</a>
        </div>
      </nav>

      <header className="hero">
        <div className="wrap hero-in">
          <div className="eyebrow"><span className="pulse" />USED BY 500+ SPORTS PHOTOGRAPHERS</div>
          <h1 className="hero-title">
            Find your best shots in <span className="grad-text">minutes,</span> not hours.
          </h1>
          <p className="hero-sub">
            Drop a full game shoot in. ClutchCull&apos;s AI locks onto the athlete, cuts blurry
            frames and duplicates, and ranks your sharpest keepers — automatically.
          </p>
          <div className="hero-cta">
            <a className="btn btn-primary" href={APP_URL}>Cull my shoot free →</a>
            <a className="btn btn-ghost" href="#how">See how it works</a>
          </div>
          <div className="hero-trust">
            <span><span className="check">✓</span> 100% free</span>
            <span><span className="check">✓</span> No sign-up</span>
            <span><span className="check">✓</span> Photos stay private</span>
          </div>

          <div className="mock">
            <div className="mock-bar"><i /><i /><i /><span className="url">clutchcull.app</span></div>
            <div className="mock-body">
              {shots.map((s, i) => (
                <div className="shot" key={i}>
                  <span className="tag">{s.tag}</span>
                  <span className="score">{s.score}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </header>

      <section className="section" id="impact">
        <div className="wrap">
          <div className="sec-head">
            <span className="sec-tag"><span className="pulse" />Live proof</span>
            <h2>Impact Dashboard</h2>
            <p>Real photographers, real shoots — a live look at how much time ClutchCull has saved.</p>
          </div>
          <div className="stats" style={{ margin: 0 }}>
            {dash.map((s) => (
              <div className="stat" key={s.lbl}>
                <div className="num">{s.num}</div>
                <div className="lbl">{s.lbl}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section" id="features">
        <div className="wrap">
          <div className="sec-head">
            <span className="sec-tag"><span className="pulse" />Why it&apos;s different</span>
            <h2>It scores the athlete — not just the frame.</h2>
            <p>Most tools measure whole-frame sharpness. ClutchCull finds your subject and judges the shot the way you would.</p>
          </div>
          <div className="feat-grid">
            {features.map((f) => (
              <div className="feat" key={f.title}>
                <div className="ic">
                  <svg viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">{f.icon}</svg>
                </div>
                <h3>{f.title}</h3>
                <p>{f.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section" id="how">
        <div className="wrap">
          <div className="sec-head">
            <span className="sec-tag"><span className="pulse" />How it works</span>
            <h2>Three steps. Done before you&apos;d finish your first coffee.</h2>
          </div>
          <div className="steps">
            {steps.map((s) => (
              <div className="step" key={s.n}>
                <div className="n">{s.n}</div>
                <h3>{s.title}</h3>
                <p>{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="wrap">
        <section className="cta">
          <div className="cta-in">
            <h2>Cull your next game in seconds.</h2>
            <p>Free. No sign-up. Built from the sideline by a student-athlete photographer.</p>
            <a className="btn" href={APP_URL}>Launch ClutchCull →</a>
          </div>
        </section>
      </div>

      <footer className="wrap">
        <div className="logo" style={{ fontSize: "1.1rem" }}>
          <span className="dot" style={{ width: 24, height: 24, fontSize: "0.8rem" }}>C</span>ClutchCull
        </div>
        <div>Built by @gec.shots · © 2026</div>
      </footer>
    </>
  );
}
