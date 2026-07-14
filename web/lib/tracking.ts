// Usage tracking — posts to the same Google Form the Streamlit app uses, so
// the shared Impact Dashboard keeps growing. Fire-and-forget, no-cors.
const FORM =
  "https://docs.google.com/forms/u/0/d/e/1FAIpQLSdE_xxiIaiHwYX9LQag1kipieTojmqEfqv1fVqwtsCKo45Mlg/formResponse";
const F = {
  event: "entry.1792514521",
  email: "entry.1250685824",
  photos: "entry.1966720800",
  exports: "entry.23206585",
  minutes: "entry.1192573871",
  session: "entry.747824885",
};

function sessionId(): string {
  if (typeof window === "undefined") return "";
  const existing = localStorage.getItem("cc_sid");
  if (existing) return existing;
  const id: string = (globalThis.crypto as any)?.randomUUID?.() || String(Date.now()) + Math.random();
  localStorage.setItem("cc_sid", id);
  return id;
}

async function post(params: Record<string, string>) {
  try {
    await fetch(FORM, {
      method: "POST",
      mode: "no-cors",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams(params).toString(),
    });
  } catch {
    /* best-effort */
  }
}

export function trackSessionStart() {
  if (typeof window === "undefined") return;
  if (sessionStorage.getItem("cc_started")) return;
  sessionStorage.setItem("cc_started", "1");
  post({ [F.event]: "session_start", [F.session]: sessionId() });
}

export function trackPhotos(n: number, email = "") {
  post({ [F.event]: "photos_processed", [F.photos]: String(n), [F.email]: email, [F.session]: sessionId() });
}

export function trackExport(minutes: number, n: number, email = "") {
  post({
    [F.event]: "export_completed", [F.exports]: "1",
    [F.minutes]: minutes.toFixed(2), [F.photos]: String(n), [F.email]: email, [F.session]: sessionId(),
  });
}

export function trackCanvas(n: number, email = "") {
  post({ [F.event]: "canvas_created", [F.exports]: "1", [F.photos]: String(n), [F.email]: email, [F.session]: sessionId() });
}

export function trackEmail(email: string) {
  post({ [F.event]: "email_provided", [F.email]: email, [F.session]: sessionId() });
}
