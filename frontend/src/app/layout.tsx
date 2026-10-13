import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import Nav from "@/components/Nav";

const plexSans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--font-plex-sans" });
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-plex-mono" });

export const metadata: Metadata = {
  title: "Listing Day · Indian IPO listing-gain predictor",
  description:
    "Predicts the listing-day gain of Indian mainboard IPOs from pre-listing subscription and market data, explained with SHAP. Educational project, not investment advice.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable}`}>
      <body className="min-h-screen">
        <div className="border-b border-line bg-warn-bg text-warn-ink">
          <p className="mx-auto max-w-6xl px-4 py-2 text-xs sm:text-[13px]">
            <strong>Educational project.</strong> Not investment advice and not SEBI-registered advice. Predictions are
            uncertain and past IPO behaviour does not guarantee future listings.
          </p>
        </div>
        <Nav />
        <main className="mx-auto max-w-6xl px-4 pb-16 pt-6 sm:pt-10">{children}</main>
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 text-xs text-muted sm:flex-row sm:justify-between">
            <p>
              Course project, UE24AM342AA4 Machine Learning with Business Data, PES University. Not investment advice;
              not SEBI-registered advice.
            </p>
            <p className="flex gap-4">
              <Link className="underline underline-offset-2 hover:text-ink" href="/about">
                Method &amp; limitations
              </Link>
              <a
                className="underline underline-offset-2 hover:text-ink"
                href="https://github.com/Aarti-panchal01/ipo-listing-predictor"
              >
                Source on GitHub
              </a>
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
