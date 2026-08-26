import type { Metadata } from "next";
import { Archivo, Inter, JetBrains_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

// Three roles, like a real tool:
//  - Archivo: headlines. Slightly condensed, editorial/athletic, has actual character.
//  - Inter: UI text. Deliberately invisible so it never competes with photos.
//  - JetBrains Mono: numbers (scores, counts, filenames). Tabular figures stop
//    column jitter and read as instrumentation rather than marketing.
const display = Archivo({ subsets: ["latin"], weight: ["600", "700", "800"], variable: "--font-display" });
const ui = Inter({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--font-ui" });
const mono = JetBrains_Mono({ subsets: ["latin"], weight: ["400", "500", "700"], variable: "--font-mono" });

// SEO: real title/description/OG so searching "ClutchCull" surfaces the site.
// Use the real deployed domain, clutchcull.app does not resolve, which broke
// Open Graph link previews when the site was shared on social.
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://gec-shots-clutch-cull.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: "ClutchCull, Shoot the Moment. Skip the Culling.",
  description:
    "ClutchCull culls a full photo shoot in minutes. AI subject detection cuts blurry frames and duplicates and ranks your sharpest keepers, sports, portraits, or events. Free, no sign-up, by Gec Shots.",
  keywords: [
    "photo culling",
    "sports photography",
    "event photography",
    "AI photo culling",
    "cull photos",
    "best photos",
    "ClutchCull",
    "Gec Shots",
  ],
  authors: [{ name: "Gec Shots" }],
  openGraph: {
    title: "ClutchCull, Shoot the moment. Skip the culling.",
    description:
      "Free AI photo culling for photographers. Cut blurry frames and duplicates, keep your sharpest shots, in minutes.",
    url: SITE_URL,
    siteName: "ClutchCull",
    images: ["/og.png"],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "ClutchCull, Shoot the Moment. Skip the Culling.",
    description: "Free AI photo culling, cut blurry frames and duplicates, keep your sharpest shots in minutes.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${ui.variable} ${mono.variable}`}>
      <body>
        {children}
        {/* Cookieless traffic analytics, visitors, referrers, devices.
            Covers the landing page too, which previously tracked nothing. */}
        <Analytics />
      </body>
    </html>
  );
}
