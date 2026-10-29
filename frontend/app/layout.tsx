import type { Metadata } from 'next';
import type { ReactNode } from 'react';

import Nav from '@/components/Nav';

import './globals.css';

export const metadata: Metadata = {
  title: 'DR Analysis — research dashboard',
  description: 'Diabetic retinopathy research dashboard: fundus classification and lesion segmentation.',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:rounded focus:bg-white focus:px-3 focus:py-2">
          Skip to content
        </a>
        <Nav />
        <main id="main" className="mx-auto max-w-6xl px-4 py-6">
          {children}
        </main>
        <footer className="mx-auto max-w-6xl px-4 pb-8 text-xs text-slate-500">
          Research application. Not a medical device. Not clinically validated. Uploaded images are processed in memory and are not stored.
        </footer>
      </body>
    </html>
  );
}
