"use client";

import { useRef, useEffect, useState, useMemo } from "react";
import { PCA } from "@/core/algorithms/pca";
import type { PCAState } from "@/core/algorithms/pca";

interface PCAVisualizationProps {
  algorithm: PCA;
  trainData?: {
    features: number[][];
    targets?: number[];
  };
}

export function PCAVisualization({
  algorithm,
  trainData,
}: PCAVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{
    x: number;
    y: number;
  } | null>(null);

  // Initialize the algorithm's state with the training data
  const pcaState = useMemo(() => {
    if (!trainData) return null;
    return algorithm.initialize(trainData.features, trainData.targets);
  }, [algorithm, trainData]);

  useEffect(() => {
    if (!canvasRef.current || !trainData || !pcaState || !pcaState.projections) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const padding = 40;
    const width = canvas.width - 2 * padding;
    const height = canvas.height - 2 * padding;

    // Find projection ranges
    let minX = pcaState.projections[0][0];
    let maxX = pcaState.projections[0][0];
    let minY = pcaState.projections[0][1];
    let maxY = pcaState.projections[0][1];

    for (const proj of pcaState.projections) {
      minX = Math.min(minX, proj[0]);
      maxX = Math.max(maxX, proj[0]);
      minY = Math.min(minY, proj[1]);
      maxY = Math.max(maxY, proj[1]);
    }

    // Add margin
    const xMargin = (maxX - minX) * 0.1 || 1;
    const yMargin = (maxY - minY) * 0.1 || 1;
    minX -= xMargin;
    maxX += xMargin;
    minY -= yMargin;
    maxY += yMargin;

    // Utility functions
    const toCanvasX = (x: number) => padding + ((x - minX) / (maxX - minX)) * width;
    const toCanvasY = (y: number) =>
      canvas.height - padding - ((y - minY) / (maxY - minY)) * height;

    // Color palette for classes (if targets provided)
    const colors = [
      "#ef4444", // red
      "#3b82f6", // blue
      "#10b981", // emerald
      "#f59e0b", // amber
      "#8b5cf6", // violet
      "#06b6d4", // cyan
      "#ec4899", // pink
      "#eab308", // lime
    ];

    const getColor = (classIdx: number) => colors[classIdx % colors.length];

    // Clear canvas
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw density background
    const densityResolution = 30;
    for (let py = 0; py < canvas.height; py += densityResolution) {
      for (let px = 0; px < canvas.width; px += densityResolution) {
        // Count nearby points
        let nearbyCount = 0;
        const x = minX + ((px - padding) / width) * (maxX - minX);
        const y = minY + ((canvas.height - padding - py) / height) * (maxY - minY);

        for (const proj of pcaState.projections) {
          const dx = proj[0] - x;
          const dy = proj[1] - y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 0.3) nearbyCount++;
        }

        if (nearbyCount > 0) {
          const intensity = Math.min(255, nearbyCount * 20);
          ctx.fillStyle = `rgba(200, 200, 200, ${intensity / 255 * 0.1})`;
          ctx.fillRect(px, py, densityResolution, densityResolution);
        }
      }
    }

    // Draw data points
    const pointRadius = 5;
    for (let i = 0; i < pcaState.projections.length; i++) {
      const [x, y] = pcaState.projections[i];
      const classIdx = trainData.targets ? trainData.targets[i] : 0;
      const canvasX = toCanvasX(x);
      const canvasY = toCanvasY(y);

      // Draw point
      ctx.fillStyle = getColor(classIdx);
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, pointRadius, 0, 2 * Math.PI);
      ctx.fill();

      // Draw border
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Highlight hovered point
      if (hoveredPoint) {
        const dist = Math.sqrt(
          Math.pow(hoveredPoint.x - canvasX, 2) +
            Math.pow(hoveredPoint.y - canvasY, 2)
        );
        if (dist < 15) {
          ctx.strokeStyle = "#000000";
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(canvasX, canvasY, pointRadius + 4, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    }

    // Draw axes with origin
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;

    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.stroke();

    // Origin crosshair
    const originX = toCanvasX(0);
    const originY = toCanvasY(0);
    if (
      originX > padding &&
      originX < canvas.width - padding &&
      originY > padding &&
      originY < canvas.height - padding
    ) {
      ctx.strokeStyle = "#9ca3af";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(originX - 5, originY);
      ctx.lineTo(originX + 5, originY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(originX, originY - 5);
      ctx.lineTo(originX, originY + 5);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw axis labels
    ctx.fillStyle = "#6b7280";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(
      `PC1 (${(pcaState.variance[0] * 100).toFixed(1)}%)`,
      canvas.width / 2,
      canvas.height - 5
    );
    ctx.save();
    ctx.translate(5, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(
      `PC2 (${(pcaState.variance[1] * 100).toFixed(1)}%)`,
      0,
      0
    );
    ctx.restore();

    // Draw ticks
    for (let i = 0; i <= 4; i++) {
      const x = minX + (i / 4) * (maxX - minX);
      const canvasX = toCanvasX(x);
      ctx.beginPath();
      ctx.moveTo(canvasX, canvas.height - padding + 5);
      ctx.lineTo(canvasX, canvas.height - padding - 5);
      ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(x.toFixed(1), canvasX, canvas.height - padding + 20);
    }

    for (let i = 0; i <= 4; i++) {
      const y = minY + (i / 4) * (maxY - minY);
      const canvasY = toCanvasY(y);
      ctx.textAlign = "right";
      ctx.fillText(y.toFixed(1), padding - 10, canvasY + 4);
    }

    // Draw variance explained legend
    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(
      `Variance explained: ${(pcaState.cumulativeVariance[1] * 100).toFixed(1)}%`,
      padding + 10,
      padding + 20
    );
  }, [algorithm, trainData, pcaState, hoveredPoint]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    setHoveredPoint({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  return (
    <div className="w-full h-full flex flex-col items-center justify-center bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <canvas
        ref={canvasRef}
        width={500}
        height={500}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 cursor-crosshair"
      />
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
        Data projected onto first 2 principal components. Gray regions show data density.
      </p>
    </div>
  );
}
