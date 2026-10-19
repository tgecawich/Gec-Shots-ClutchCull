import { getImpact, shots, lowShots, APP_URL, RECENT_WINDOW_DAYS } from "@/lib/impact";
import s from "./home.module.css";

// Shown only if the sheet fetch fails, so the page never renders an empty
// dashboard. Deliberately vague rather than a stale hardcoded number: a wrong
// figure is worse than no figure on a counter that claims to be live.
const FALLBACK_LINE = "Built by a photographer, for photographers";

const proof = [
  {
    n: "01",
    title: "It scores the athlete",
    body: "Most tools measure the whole frame, so a blurry player in front of a crisp scoreboard scores well. ClutchCull locates your subject first, then judges that.",
  },
  {
    n: "02",
    title: "Nothing is ever deleted",
    body: "Every rejected frame stays one tap from being rescued. The software ranks; the call is always yours.",
  },
  {
    n: "03",
    title: "Reads RAW directly",
    body: "It pulls the preview your camera already embedded, so a 40MB RAW never has to be uploaded or converted first.",
  },
];

const TICKER = (
  <span>
    RAW FILES SUPPORTED <i>◆</i> CR3 · NEF · ARW · DNG <i>◆</i> LIGHTROOM XMP EXPORT <i>◆</i>{" "}
    PHOTOS NEVER LEAVE YOUR DEVICE <i>◆</i> NO SIGN-UP <i>◆</i> FREE FOREVER <i>◆</i> BUILT ON THE SIDELINE <i>◆</i>{" "}
  </span>
);

export default async function Home() {
  const impact = await getImpact();
  const heroShots = shots.slice(0, 4);

  return (
    <div className={s.page}>
      <nav className={s.nav}>
        <div className={`${s.wrap} ${s.navIn}`}>
          <a className={s.brand} href="/">
            <span className={s.mark}>C</span>
            <span>
              <span className={s.brandName}>ClutchCull</span>
              <span className={s.brandSub}>BY GEC SHOTS</span>
            </span>
          </a>
          <div className={s.navLinks}>
            <a href="#proof">Features</a>
            <a href="#shoot">How it works</a>
            <a href={APP_URL}>Canvas</a>
            <a href="#impact">Impact</a>
          </div>
          <a className={s.btn} href={APP_URL}>Launch app →</a>
        </div>
      </nav>

      <header className={s.hero}>
        <div className={s.stripes}><i /><i /><i /></div>
        <div className={`${s.wrap} ${s.heroWrap} ${s.heroGrid}`}>
          <div>
            {/* The live figure is the FIRST line of text on the page, which is
                how the original design did it. On a phone the hero stacks and
                the counters below sit ~245px under the fold, so without this
                the proof is a scroll away for most of the audience. */}
            <div className={s.eyebrow}>
              <i />
              {impact
                ? `${Math.round(impact.photos).toLocaleString()} photos culled by ${impact.people.toLocaleString()} photographers`
                : FALLBACK_LINE}
            </div>
            <h1 style={{ marginTop: 18 }}>
              <span className={s.l1}>SHOOT THE MOMENT.</span>
              <span className={s.l2}>SKIP THE CULLING.</span>
            </h1>
            <p className={s.sub}>
              Drop a full shoot in. ClutchCull finds your subject, cuts blurry frames and
              duplicate bursts, then ranks your sharpest keepers.
            </p>
            <div className={s.cta}>
              <a className={`${s.btn} ${s.btnLg}`} href={APP_URL}>Cull my shoot free →</a>
              <a className={`${s.btn} ${s.btnLg} ${s.btnGhost}`} href="#shoot">See how it works</a>
            </div>
            <div className={s.trust}>
              Free forever &nbsp;·&nbsp; No sign-up &nbsp;·&nbsp; Photos stay on your device
            </div>
          </div>

          <div className={s.heroShots}>
            {heroShots.map((p, i) => (
              <div className={`${s.hs} ${i < 2 ? s.keep : ""}`} key={p.src}>
                <img src={p.src} alt={p.alt} loading={i < 2 ? "eager" : "lazy"} />
                <span className={s.hsScore}>{p.score}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Impact sits INSIDE the hero, not in a section below it. At the fold
            is not the same as on screen: on a 1366x768 laptop the browser
            chrome leaves roughly 650px, and anything below that is a scroll the
            visitor may never make. Every figure is summed from the live log. */}
        <div className={`${s.wrap} ${s.heroWrap} ${s.impactIn}`} id="impact">
          <div className={s.impactLab}><span className={s.pulse} />Live impact · updating in real time</div>
          {impact ? (
            <>
              <div className={s.bigStats}>
                <div className={s.bigStat}>
                  <div className={s.bigN}>{Math.round(impact.photos).toLocaleString()}</div>
                  <div className={s.bigL}>Photos culled</div>
                </div>
                <div className={s.bigStat}>
                  <div className={s.bigN}>{impact.people.toLocaleString()}</div>
                  <div className={s.bigL}>Photographers</div>
                </div>
                <div className={s.bigStat}>
                  <div className={s.bigN}>{Math.round(impact.exports).toLocaleString()}</div>
                  <div className={s.bigL}>Exports</div>
                </div>
                <div className={s.bigStat}>
                  <div className={s.bigN}>{Math.round(impact.hours).toLocaleString()}</div>
                  <div className={s.bigL}>Hours saved</div>
                </div>
              </div>
              {impact.recentPhotos > 0 && (
                <div className={s.momentum}>
                  <span className={s.momTag}>Last {RECENT_WINDOW_DAYS} days</span>
                  <span>
                    <b>{Math.round(impact.recentPhotos).toLocaleString()}</b> photos culled
                    {impact.recentPeople > 0 && (
                      <> and <b>{impact.recentPeople.toLocaleString()}</b> new photographers</>
                    )}
                  </span>
                </div>
              )}
            </>
          ) : (
            <div className={s.bigL} style={{ marginTop: 0 }}>{FALLBACK_LINE}</div>
          )}
        </div>
      </header>

      <div className={s.ticker}>
        <div className={s.run}>
          {TICKER}
          {/* duplicated so the marquee loops without a gap */}
          {TICKER}
        </div>
      </div>

      <section className={s.sec} id="shoot">
        <div className={s.wrap}>
          <div className={s.lab}><span /><span className={s.labT}>One 33-frame shoot, as the engine ranked it</span></div>

          <div className={s.rankHead}>
            <span className={`${s.rk} ${s.rkTop}`}>Ranked 1 – 3</span>
            <span className={s.rkNote}>kept, sorted by score</span>
          </div>
          <div className={s.shots}>
            {shots.slice(0, 3).map((p) => (
              <div className={s.shot} key={p.src}>
                <img src={p.src} alt={p.alt} loading="lazy" />
                <span className={s.tg}>{p.tag}</span>
                <span className={s.sc}>{p.score}</span>
              </div>
            ))}
          </div>

          <div className={s.rankHead} style={{ marginTop: 40 }}>
            <span className={`${s.rk} ${s.rkLow}`}>Ranked 31 – 33</span>
            <span className={s.rkNote}>same shoot, bottom of the list</span>
          </div>
          <div className={s.shots}>
            {lowShots.map((p) => (
              <div className={`${s.shot} ${s.dim}`} key={p.src}>
                <img src={p.src} alt={p.alt} loading="lazy" />
                <span className={s.tg}>{p.tag}</span>
                <span className={`${s.sc} ${s.scLow}`}>{p.score}</span>
              </div>
            ))}
          </div>

          <div className={s.capRow}>
            <span>
              Nothing here is cherry-picked: these are positions 1–3 and 31–33 of the same 33
              frames. Small, backlit or static subjects sink on their own.
            </span>
            <span>EG Avengers v. La Salle</span>
          </div>

        </div>
      </section>

      <section className={`${s.sec} ${s.secAlt}`} id="proof">
        <div className={s.wrap}>
          <div className={s.lab}><span /><span className={s.labT}>Why it beats a sharpness filter</span></div>
          <div className={s.proofGrid}>
            {proof.map((f) => (
              <div key={f.n}>
                <div className={s.pnum}>{f.n}</div>
                <h3>{f.title}</h3>
                <p>{f.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className={s.band}>
        <div className={`${s.wrap} ${s.bandWrap}`}>
          <h2>
            CULL YOUR NEXT GAME.
            <br />
            <em>IN MINUTES, NOT HOURS.</em>
          </h2>
          <p>
            Free, no sign-up, and your originals never leave your device. Built on the sideline
            by a photographer who got tired of culling at 1am.
          </p>
          <a className={`${s.btn} ${s.btnLg}`} style={{ marginTop: 32 }} href={APP_URL}>
            Launch ClutchCull →
          </a>
        </div>
      </section>

      <footer className={s.footer}>
        <div className={`${s.wrap} ${s.footerIn}`}>
          <span>ClutchCull · Built by @gec.shots</span>
          <span>
            <a href="/classic">Original design</a> &nbsp;·&nbsp; © 2026
          </span>
        </div>
      </footer>
    </div>
  );
}
