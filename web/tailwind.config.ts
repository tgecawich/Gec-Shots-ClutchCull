import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#08060f",
        panel: "#12102a",
        ink: "#f8f4ff",
        muted: "#b7aacb",
        brand: {
          orange: "#ff8a2a",
          gold: "#ffc15a",
          purple: "#7b3cff",
          pink: "#ff4ecd",
        },
      },
      fontFamily: {
        sans: ["var(--font-jakarta)", "system-ui", "sans-serif"],
      },
      backgroundImage: {
        brand: "linear-gradient(135deg,#7b3cff 0%,#ff4ecd 50%,#ff8a2a 100%)",
      },
      borderColor: {
        hair: "rgba(255,255,255,0.10)",
      },
    },
  },
  plugins: [],
};
export default config;
