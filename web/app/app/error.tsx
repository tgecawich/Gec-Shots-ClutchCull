"use client";

import { useEffect } from "react";

// Route-level error boundary. Without this, any uncaught client error shows
// Next.js's bare "Application error: a client-side exception has occurred",
// which loses the user's whole shoot with no explanation or way back.
export default function AppError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error("ClutchCull app error:", error);
  }, [error]);

  return (
    <div className="wrap" style={{ padding: "80px 24px", maxWidth: 640 }}>
      <h1 className="app-h1" style={{ fontSize: "1.9rem" }}>Something broke on this screen</h1>
      <p className="app-lead" style={{ marginBottom: 22 }}>
        Your photos are safe — nothing was uploaded or changed. This is usually a very large
        shoot straining the browser. Try again, and if it keeps happening, cull in two
        smaller batches.
      </p>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <button className="btn btn-primary" onClick={reset}>Try again</button>
        <a className="btn btn-ghost" href="/app">Start a new cull</a>
      </div>
      {error?.message && (
        <p style={{ marginTop: 26, color: "var(--muted)", fontSize: "0.85rem" }}>
          Details: {error.message}
        </p>
      )}
    </div>
  );
}
