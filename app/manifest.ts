import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    id: "/",
    name: "DALMUTI",
    short_name: "DALMUTI",
    description:
      "낮은 숫자로 계급을 뒤집는 실시간 카드 게임. 빠른 대전 또는 친구들과 온라인 대전을 즐기세요.",
    start_url: "/",
    scope: "/",
    display: "standalone",
    background_color: "#000000",
    theme_color: "#18070c",
    lang: "ko",
    categories: ["games", "entertainment"],
    icons: [
      {
        src: "/pwa/icon-v2-192.png",
        sizes: "192x192",
        type: "image/png",
        purpose: "any",
      },
      {
        src: "/pwa/icon-v2-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "any",
      },
      {
        src: "/pwa/icon-v2-1024.png",
        sizes: "1024x1024",
        type: "image/png",
        purpose: "any",
      },
      {
        src: "/pwa/icon-maskable-v2-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
    shortcuts: [
      {
        name: "빠른 대전",
        short_name: "빠른 대전",
        description: "4~10인 빠른 대전을 시작합니다.",
        url: "/",
        icons: [
          {
            src: "/pwa/icon-v2-192.png",
            sizes: "192x192",
            type: "image/png",
          },
        ],
      },
      {
        name: "온라인 대전",
        short_name: "온라인",
        description: "방을 만들거나 초대 코드로 참가합니다.",
        url: "/online",
        icons: [
          {
            src: "/pwa/icon-v2-192.png",
            sizes: "192x192",
            type: "image/png",
          },
        ],
      },
    ],
  };
}
