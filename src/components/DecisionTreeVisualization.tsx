"use client";

import { useRef, useEffect, useState, useMemo } from "react";
import { DecisionTree } from "@/core/algorithms/decision-tree";
import type { DecisionTreeState } from "@/core/algorithms/decision-tree";

interface DecisionTreeVisualizationProps {
  algorithm: DecisionTree;
  trainData?: {
    features: number[][];
    targets: number[];
  };
}

export function DecisionTreeVisualization({
  algorithm,
  trainData,
}: DecisionTreeVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{
    x: number;
    y: number;
  } | null>(null);

  // Initialize the algorithm's state with the training data
  const treeState = useMemo(() => {
    if (!trainData) return null;
    return algorithm.initialize(trainData.features, trainData.targets);
  }, [algorithm, trainData]);

  useEffect(() => {
    if (!canvasRef.current || !trainData || !treeState) return;

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

    // Color palette for classes
    const colors = [
      "#ef4444", // red
      "#3b82f6", // blue
      "#10b981", // emerald
      "#f59e0b", // amber
      "#8b5cf6", // violet
    ];

    const getColor = (classIdx: number) => colors[classIdx % colors.length];

    // Clear canvas
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw decision regions (mesh grid)
    const meshResolution = 50;
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;

    for (let py = 0; py < canvas.height; py++) {
      for (let px = 0; px < canvas.width; px++) {
        const x = toFeatureX(px);
        const y = toFeatureY(py);

        // Predict class at this point using the tree
        const input = [x, y];
        const { result: prediction } = algorithm.forward(treeState, input);
        const color = getColor(prediction);

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
        data[idx + 3] = 40; // Low opacity mesh
      }
    }

    ctx.putImageData(imageData, 0, 0);

    // Draw training data points
    const pointRadius = 5;
    for (let i = 0; i < trainData.features.length; i++) {
      const [x, y] = trainData.features[i];
      const classIdx = trainData.targets[i];
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
  }, [algorithm, trainData, treeState, hoveredPoint]);

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
        Decision regions colored by class. Points show training data.
      </p>
    </div>
  );
}
