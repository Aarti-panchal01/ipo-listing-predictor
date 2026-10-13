"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Predictor" },
  { href: "/past", label: "Past IPOs" },
  { href: "/insights", label: "Insights" },
  { href: "/about", label: "About" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="flex items-baseline gap-2">
          <span className="font-mono text-lg font-semibold tracking-tight">LISTING&nbsp;DAY</span>
          <span className="hidden text-xs text-muted sm:inline">Indian IPO listing-gain predictor</span>
        </Link>
        <nav aria-label="Main" className="-mx-1 flex gap-1 overflow-x-auto">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? "page" : undefined}
                className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors ${
                  active ? "bg-ink text-page" : "text-ink-2 hover:bg-surface-2 hover:text-ink"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
