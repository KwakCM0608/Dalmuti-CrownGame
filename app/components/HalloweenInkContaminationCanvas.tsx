"use client";

import { useEffect, useRef } from "react";

const INK_BLOOM_MASK = "/themes/halloween/ink-impact-bloom-mask-v1.png";
const INK_FIELD_TEXTURE = "/themes/halloween/ink-wash-field-texture-v2.webp";
// The CSS drop reaches --ink-impact-y at 76% of its 1.18s fall animation.
// Begin the bloom on that exact frame so impact and absorption are continuous.
const DROP_IMPACT_MS = 897;
const DROP_STAGGER_MS = 90;
const SOAK_DURATION_MS = 2200;

const IMPACTS = Array.from({ length: 7 }, (_, index) => ({
  x: (8 + ((index * 19) % 86)) / 100,
  y: (24 + ((index * 17) % 48)) / 100,
  delay: index * DROP_STAGGER_MS,
  rotation: ((-18 + ((index * 37) % 43)) * Math.PI) / 180,
  stretch: 0.88 + (index % 3) * 0.11,
}));

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load ${src}`));
    image.src = src;
  });
}

function smoothstep(value: number): number {
  return value * value * (3 - 2 * value);
}

function drawCover(
  context: CanvasRenderingContext2D,
  image: HTMLImageElement,
  width: number,
  height: number,
) {
  const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight);
  const drawWidth = image.naturalWidth * scale;
  const drawHeight = image.naturalHeight * scale;
  context.drawImage(
    image,
    (width - drawWidth) / 2,
    (height - drawHeight) / 2,
    drawWidth,
    drawHeight,
  );
}

export default function HalloweenInkContaminationCanvas({
  className,
}: {
  className: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const maskCanvas = document.createElement("canvas");
    const maskContext = maskCanvas.getContext("2d");
    if (!maskContext) return;

    let animationFrame = 0;
    let cancelled = false;
    let width = 0;
    let height = 0;
    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    const completeAt =
      DROP_IMPACT_MS +
      (IMPACTS.length - 1) * DROP_STAGGER_MS +
      SOAK_DURATION_MS;
    const startedAt = performance.now() - (reducedMotion ? completeAt : 0);

    const syncCanvasSize = () => {
      const rect = canvas.getBoundingClientRect();
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.25);
      const nextWidth = Math.max(1, Math.round(rect.width * pixelRatio));
      const nextHeight = Math.max(1, Math.round(rect.height * pixelRatio));
      if (nextWidth === width && nextHeight === height) return;
      width = nextWidth;
      height = nextHeight;
      canvas.width = width;
      canvas.height = height;
      maskCanvas.width = width;
      maskCanvas.height = height;
    };

    const resizeObserver = new ResizeObserver(syncCanvasSize);
    resizeObserver.observe(canvas);
    syncCanvasSize();

    void Promise.all([loadImage(INK_BLOOM_MASK), loadImage(INK_FIELD_TEXTURE)])
      .then(([bloomMask, fieldTexture]) => {
        const render = (now: number) => {
          if (cancelled) return;
          syncCanvasSize();

          const elapsed = now - startedAt;
          const diagonal = Math.hypot(width, height);
          let allSettled = true;

          maskContext.clearRect(0, 0, width, height);
          for (const impact of IMPACTS) {
            const localTime = elapsed - DROP_IMPACT_MS - impact.delay;
            if (localTime <= 0) {
              allSettled = false;
              continue;
            }

            const progress = Math.min(localTime / SOAK_DURATION_MS, 1);
            if (progress < 1) allSettled = false;
            const eased = smoothstep(progress);
            const bloomSize =
              diagonal * (0.025 + 0.32 * progress + 4.44 * eased ** 2.4);

            maskContext.save();
            maskContext.globalAlpha = 1;
            maskContext.translate(impact.x * width, impact.y * height);
            maskContext.rotate(impact.rotation);
            maskContext.scale(impact.stretch, 1);
            maskContext.drawImage(
              bloomMask,
              -bloomSize / 2,
              -bloomSize / 2,
              bloomSize,
              bloomSize,
            );
            maskContext.restore();
          }

          context.clearRect(0, 0, width, height);
          context.globalCompositeOperation = "source-over";
          context.drawImage(maskCanvas, 0, 0);
          context.globalCompositeOperation = "source-in";
          drawCover(context, fieldTexture, width, height);
          context.globalCompositeOperation = "source-atop";
          context.fillStyle = "rgba(7, 8, 11, 0.18)";
          context.fillRect(0, 0, width, height);
          context.globalCompositeOperation = "source-over";

          if (!allSettled) animationFrame = requestAnimationFrame(render);
        };

        animationFrame = requestAnimationFrame(render);
      })
      .catch(() => {
        // Reveal the authoritative final field rather than leaving the gray
        // transition surface mounted if an optional VFX asset cannot decode.
        canvas.parentElement?.style.setProperty("display", "none");
      });

    return () => {
      cancelled = true;
      resizeObserver.disconnect();
      cancelAnimationFrame(animationFrame);
    };
  }, []);

  return <canvas ref={canvasRef} className={className} aria-hidden="true" />;
}
