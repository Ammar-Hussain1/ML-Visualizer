import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ML-Visualizer | Interactive Machine Learning Concepts",
  description: "Explore and visualize core machine learning algorithms including regression, classification, and neural networks in a beautiful interactive environment.",
  openGraph: {
    title: "ML-Visualizer",
    description: "Interactive Machine Learning Visualization Tool",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "ML-Visualizer",
    description: "Explore machine learning algorithms visually.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
