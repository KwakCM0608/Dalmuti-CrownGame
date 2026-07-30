import type { Metadata, Viewport } from "next";
import { headers } from "next/headers";
import { PwaLifecycle } from "@/app/components/PwaLifecycle";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  viewportFit: "cover",
  themeColor: "#18070c",
  colorScheme: "dark",
};

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("host") ?? "localhost:3000";
  const protocol =
    requestHeaders.get("x-forwarded-proto") ??
    (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const title = "DALMUTI";
  const description =
    "낮은 숫자로 계급을 뒤집는 달무티 웹게임. 혼자 연습하거나 초대 코드로 4~8명이 함께 플레이하세요.";

  return {
    title,
    description,
    applicationName: "DALMUTI",
    manifest: "/manifest.webmanifest",
    icons: {
      icon: "/brand-dalmuti-crown.png",
      shortcut: "/brand-dalmuti-crown.png",
      apple: [
        { url: "/pwa/apple-touch-icon.png", sizes: "180x180", type: "image/png" },
      ],
    },
    appleWebApp: {
      capable: true,
      statusBarStyle: "black-translucent",
      title: "DALMUTI",
    },
    formatDetection: {
      telephone: false,
    },
    other: {
      "mobile-web-app-capable": "yes",
    },
    openGraph: {
      title,
      description,
      type: "website",
      images: [{ url: `${origin}/og.png`, width: 1792, height: 896 }],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [`${origin}/og.png`],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>
        {children}
        <PwaLifecycle />
      </body>
    </html>
  );
}
