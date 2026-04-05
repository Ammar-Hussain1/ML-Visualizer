"use client";

import React, { useMemo } from "react";
import { Vector, Matrix } from "@/core/types";
import { DistanceMetrics } from "@/core/math/distance-metrics";

/**
 * KNN Visualization: Shows data points and decision regions
 */

interface KNNVisualizationProps {
  trainData?: { features: Matrix; targets: Vector };
  k?: number;
  metric?: "euclidean" | "manhattan" | "cosine";
  highlightPoint?: Vector;
}

export function KNNVisualization({
  trainData,
  k = 3,
  metric = "euclidean",
  highlightPoint,
}: KNNVisualizationProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  // Generate decision regions
  const meshData = useMemo(() => {
    if (!trainData || !canvasRef.current) return null;

    const canvas = canvasRef.current;
    const width = canvas.width;
    const height = canvas.height;

    // Find data bounds
    const xs = trainData.features.map((f) => f[0]);
    const ys = trainData.features.map((f) => f[1]);
    const xMin = Math.min(...xs) - 1;
    const xMax = Math.max(...xs) + 1;
    const yMin = Math.min(...ys) - 1;
    const yMax = Math.max(...ys) + 1;

    // Create mesh for decision boundary
    const step = 0.2;
    const meshPoints: { x: number; y: number; class: number }[] = [];

    for (let x = xMin; x <= xMax; x += step) {
      for (let y = yMin; y <= yMax; y += step) {
        // Find k nearest neighbors
        const distances = trainData.features.map((sample, idx) => ({
          dist: computeDistance([x, y], sample, metric),
          label: trainData.targets[idx],
        }));

        const nearest = distances
          .sort((a, b) => a.dist - b.dist)
          .slice(0, k);

        const votes = nearest.reduce(
          (acc, n) => {
            acc[n.label] = (acc[n.label] || 0) + 1;
            return acc;
          },
          {} as Record<number, number>
        );

        const predictedClass = Object.entries(votes).sort(
          (a, b) => b[1] - a[1]
        )[0][0];

        meshPoints.push({
          x,
          y,
          class: parseInt(predictedClass),
        });
      }
    }

    return { meshPoints, xMin, xMax, yMin, yMax };
  }, [trainData, k, metric]);

  React.useEffect(() => {
    if (!canvasRef.current || !meshData || !trainData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { xMin, xMax, yMin, yMax, meshPoints } = meshData;
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = "#f1f5f9";
    ctx.fillRect(0, 0, width, height);

    // Draw decision regions
    const colors = ["#93c5fd", "#fca5a5", "#86efac", "#fcd34d"];
    for (const point of meshPoints) {
      const px = ((point.x - xMin) / (xMax - xMin)) * width;
      const py = height - ((point.y - yMin) / (yMax - yMin)) * height;

      ctx.fillStyle = colors[point.class % colors.length];
      ctx.globalAlpha = 0.2;
      ctx.fillRect(px - 3, py - 3, 6, 6);
    }

    ctx.globalAlpha = 1;

    // Draw training points
    for (let i = 0; i < trainData.features.length; i++) {
      const f = trainData.features[i];
      const px = ((f[0] - xMin) / (xMax - xMin)) * width;
      const py = height - ((f[1] - yMin) / (yMax - yMin)) * height;

      ctx.fillStyle = colors[trainData.targets[i] % colors.length];
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Draw highlight point and its neighbors
    if (highlightPoint && highlightPoint.length >= 2) {
      const px = ((highlightPoint[0] - xMin) / (xMax - xMin)) * width;
      const py =
        height - ((highlightPoint[1] - yMin) / (yMax - yMin)) * height;

      // Draw k nearest neighbors circles
      const distances = trainData.features.map((sample, idx) => ({
        dist: computeDistance(highlightPoint, sample, metric),
        index: idx,
      }));

      const nearest = distances
        .sort((a, b) => a.dist - b.dist)
        .slice(0, k);

      const maxDist = nearest[nearest.length - 1].dist;

      for (const n of nearest) {
        const f = trainData.features[n.index];
        const nx = ((f[0] - xMin) / (xMax - xMin)) * width;
        const ny = height - ((f[1] - yMin) / (yMax - yMin)) * height;

        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(nx, ny);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Highlight the query point with a star
      ctx.fillStyle = "#f59e0b";
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height);
    ctx.lineTo(width, height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, height);
    ctx.stroke();
  }, [meshData, trainData, k, metric, highlightPoint]);

  return (
    <div className="w-full h-full flex flex-col items-center justify-center bg-card rounded-lg border border-border p-4">
      <h3 className="text-sm font-semibold mb-4 text-foreground">
        KNN Decision Regions (k={k}, {metric})
      </h3>
      <canvas
        ref={canvasRef}
        width={500}
        height={400}
        className="bg-background rounded border border-border/50"
      />
      <div className="mt-4 text-xs text-foreground/70 space-y-1">
        <p>
          <strong>Colored regions:</strong> Predicted class areas
        </p>
        <p>
          <strong>Dots:</strong> Training data points
        </p>
        <p>
          <strong>K:</strong> {k} nearest neighbors
        </p>
      </div>
    </div>
  );
}

/**
 * Helper: compute distance based on metric
 */
function computeDistance(
  p1: Vector,
  p2: Vector,
  metric: string
): number {
  switch (metric) {
    case "manhattan":
      return DistanceMetrics.manhattan(p1, p2);
    case "cosine":
      return DistanceMetrics.cosine(p1, p2);
    case "euclidean":
    default:
      return DistanceMetrics.euclidean(p1, p2);
  }
}
