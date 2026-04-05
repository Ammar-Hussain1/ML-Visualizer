"use client";

import { useRef, useEffect, useState, useMemo } from "react";
import { KMeans } from "@/core/algorithms/kmeans";
import type { KMeansState } from "@/core/algorithms/kmeans";

interface KMeansVisualizationProps {
  algorithm: KMeans;
  trainData?: {
    features: number[][];
    targets?: number[];
  };
}

export function KMeansVisualization({
  algorithm,
  trainData,
}: KMeansVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{
    x: number;
    y: number;
  } | null>(null);

  // Initialize the algorithm's state with the training data
  const kmState = useMemo(() => {
    if (!trainData) return null;
    return algorithm.initialize(trainData.features);
  }, [algorithm, trainData]);

  useEffect(() => {
    if (!canvasRef.current || !trainData || !kmState) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const padding = 40;
    const width = canvas.width - 2 * padding;
    const height = canvas.height - 2 * padding;

    // Find feature ranges
    let minX = trainData.features[0][0];
    let maxX = trainData.features[0][0];
    let minY = trainData.features[0][1];
    let maxY = trainData.features[0][1];

    for (const features of trainData.features) {
      minX = Math.min(minX, features[0]);
      maxX = Math.max(maxX, features[0]);
      minY = Math.min(minY, features[1]);
      maxY = Math.max(maxY, features[1]);
    }

    // Add margin
    const xMargin = (maxX - minX) * 0.1;
    const yMargin = (maxY - minY) * 0.1;
    minX -= xMargin;
    maxX += xMargin;
    minY -= yMargin;
    maxY += yMargin;

    // Utility functions
    const toCanvasX = (x: number) => padding + ((x - minX) / (maxX - minX)) * width;
    const toCanvasY = (y: number) =>
      canvas.height - padding - ((y - minY) / (maxY - minY)) * height;
    const toFeatureX = (canvasX: number) => minX + ((canvasX - padding) / width) * (maxX - minX);
    const toFeatureY = (canvasY: number) => minY + ((canvas.height - padding - canvasY) / height) * (maxY - minY);

    // Color palette for clusters
    const clusterColors = [
      "#ef4444", // red
      "#3b82f6", // blue
      "#10b981", // emerald
      "#f59e0b", // amber
      "#8b5cf6", // violet
      "#06b6d4", // cyan
      "#ec4899", // pink
      "#eab308", // lime
    ];

    const getClusterColor = (clusterIdx: number) => clusterColors[clusterIdx % clusterColors.length];

    // Clear canvas
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw Voronoi-like decision regions using mesh
    const meshResolution = 50;
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;

    for (let py = 0; py < canvas.height; py++) {
      for (let px = 0; px < canvas.width; px++) {
        const x = toFeatureX(px);
        const y = toFeatureY(py);

        // Find nearest centroid
        let minDist = Infinity;
        let nearestCluster = 0;

        for (let j = 0; j < kmState.centroids.length; j++) {
          const dx = x - kmState.centroids[j][0];
          const dy = y - kmState.centroids[j][1];
          const dist = Math.sqrt(dx * dx + dy * dy);
          
          if (dist < minDist) {
            minDist = dist;
            nearestCluster = j;
          }
        }

        const color = getClusterColor(nearestCluster);

        // Parse RGB from hex
        const hex = color.replace("#", "");
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);

        // Set pixel with reduced opacity for mesh effect
        const idx = (py * canvas.width + px) * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 30; // Low opacity mesh
      }
    }

    ctx.putImageData(imageData, 0, 0);

    // Draw data points (colored by cluster assignment)
    const pointRadius = 5;
    for (let i = 0; i < trainData.features.length; i++) {
      const [x, y] = trainData.features[i];
      const clusterIdx = kmState.assignments[i];
      const canvasX = toCanvasX(x);
      const canvasY = toCanvasY(y);

      // Draw point
      ctx.fillStyle = getClusterColor(clusterIdx);
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

    // Draw centroids
    for (let j = 0; j < kmState.centroids.length; j++) {
      const [cx, cy] = kmState.centroids[j];
      const canvasX = toCanvasX(cx);
      const canvasY = toCanvasY(cy);

      // Draw centroid as a star/cross
      const size = 10;
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 2.5;

      // Draw cross
      ctx.beginPath();
      ctx.moveTo(canvasX - size, canvasY);
      ctx.lineTo(canvasX + size, canvasY);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(canvasX, canvasY - size);
      ctx.lineTo(canvasX, canvasY + size);
      ctx.stroke();

      // Draw circle around centroid
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 12, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = "#6b7280";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Feature 1", canvas.width / 2, canvas.height - 5);
    ctx.save();
    ctx.translate(5, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Feature 2", 0, 0);
    ctx.restore();

    // Draw ticks
    for (let i = 0; i <= 4; i++) {
      const x = minX + (i / 4) * (maxX - minX);
      const canvasX = toCanvasX(x);
      ctx.beginPath();
      ctx.moveTo(canvasX, canvas.height - padding + 5);
      ctx.lineTo(canvasX, canvas.height - padding - 5);
      ctx.stroke();
      ctx.fillText(x.toFixed(1), canvasX, canvas.height - padding + 20);
    }

    for (let i = 0; i <= 4; i++) {
      const y = minY + (i / 4) * (maxY - minY);
      const canvasY = toCanvasY(y);
      ctx.textAlign = "right";
      ctx.fillText(y.toFixed(1), padding - 10, canvasY + 4);
    }
  }, [algorithm, trainData, kmState, hoveredPoint]);

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
        Cluster regions colored by assignments. X marks show centroids.
      </p>
    </div>
  );
}
