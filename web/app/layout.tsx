import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-jakarta",
});

// SEO: real title/description/OG so searching "ClutchCull" surfaces the site.
export const metadata: Metadata = {
  metadataBase: new URL("https://clutchcull.app"),
  title: "ClutchCull — Free AI Photo Culling for Sports Photographers",
  description:
    "ClutchCull finds your best sports photos in minutes. AI subject detection cuts blurry frames and duplicates and ranks your sharpest keepers. Free, no sign-up, by Gec Shots.",
  keywords: [
    "photo culling",
    "sports photography",
    "AI photo culling",
    "cull photos",
    "best sports photos",
    "ClutchCull",
    "Gec Shots",
  ],
  authors: [{ name: "Gec Shots" }],
  openGraph: {
    title: "ClutchCull — Find your best shots in minutes, not hours",
    description:
      "Free AI photo culling for sports photographers. Cut blurry frames and duplicates, keep your sharpest shots.",
    url: "https://clutchcull.app",
    siteName: "ClutchCull",
    images: ["/og.png"],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "ClutchCull — Free AI Photo Culling for Sports Photographers",
    description: "Find your best shots in minutes, not hours.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={jakarta.variable}>
      <body>{children}</body>
    </html>
  );
}
