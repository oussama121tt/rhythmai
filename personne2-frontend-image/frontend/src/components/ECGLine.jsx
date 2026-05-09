import React, { useRef, useEffect } from "react";
import { C } from "../utils/constants";

export default function ECGLine({ width = 320, height = 40, color = C.accent }) {
  const ref = useRef(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    let t = 0;
    let raf;

    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (let x = 0; x <= width; x++) {
        let y = height / 2;
        const mod = ((x + t * 40) % 80) / 80;
        if (mod > 0.3 && mod < 0.35) y -= height * 0.6;
        else if (mod > 0.35 && mod < 0.4) y += height * 0.3;
        else if (mod > 0.4 && mod < 0.45) y -= height * 0.1;
        y += Math.sin((x / width) * 4 * Math.PI + t) * 2;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      t += 0.02;
      raf = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(raf);
  }, [width, height, color]);

  return <canvas ref={ref} width={width} height={height} style={{ display: "block" }} />;
}
