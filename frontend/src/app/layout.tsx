import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "LLM Engine Dashboard",
  description: "High-Performance LLM Serving & Telemetry",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 antialiased font-sans">
        {children}
      </body>
    </html>
  );
}