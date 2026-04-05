"use client";

import React from "react";
import { LayerConfig } from "@/core/algorithms/neural-network";

/**
 * Neural Network Visualization: Display network architecture
 */

interface NeuralNetworkVisualizationProps {
  layers: LayerConfig[];
  weights?: number[][][]; // weights[l][i] = weights from neuron i in layer l to layer l+1
  active?: { layer: number; neuron: number } | null;
}

export function NeuralNetworkVisualization({
  layers,
  weights,
  active,
}: NeuralNetworkVisualizationProps) {
  // Input layer + hidden layers + output layer
  const allLayers = [
    { units: layers[0]?.units ?? 2, activation: "input" as const },
    ...layers,
  ];

  const svgWidth = 600;
  const svgHeight = 400;
  const padding = 40;
  const neuronRadius = 6;
  const layerSpacing = (svgWidth - 2 * padding) / (allLayers.length - 1);
  const maxNeurons = Math.max(...allLayers.map((l) => l.units));
  const neuronSpacing = (svgHeight - 2 * padding) / Math.max(maxNeurons - 1, 1);

  const getNeuronPosition = (layer: number, neuron: number) => {
    const x = padding + layer * layerSpacing;
    const layerHeight = (allLayers[layer].units - 1) * neuronSpacing;
    const offset = Math.max(0, (svgHeight - 2 * padding - layerHeight) / 2);
    const y = padding + offset + neuron * neuronSpacing;
    return { x, y };
  };

  return (
    <div className="w-full h-full flex flex-col items-center justify-center bg-card rounded-lg border border-border p-4">
      <h3 className="text-sm font-semibold mb-4 text-foreground">
        Network Architecture
      </h3>

      <svg
        width={svgWidth}
        height={svgHeight}
        className="bg-background rounded border border-border/50"
      >
        {/* Draw connections */}
        {weights &&
          weights.map((layerWeights, l) =>
            layerWeights.map((neuronWeights, fromIdx) =>
              neuronWeights.map((weight, toIdx) => {
                const from = getNeuronPosition(l, fromIdx);
                const to = getNeuronPosition(l + 1, toIdx);
                const opacity = Math.min(Math.abs(weight), 1);
                const color = weight > 0 ? "#3b82f6" : "#ef4444";

                return (
                  <line
                    key={`weight-${l}-${fromIdx}-${toIdx}`}
                    x1={from.x}
                    y1={from.y}
                    x2={to.x}
                    y2={to.y}
                    stroke={color}
                    strokeWidth="1"
                    opacity={opacity * 0.3}
                  />
                );
              })
            )
          )}

        {/* Draw neurons */}
        {allLayers.map((layer, l) =>
          Array(layer.units)
            .fill(0)
            .map((_, n) => {
              const pos = getNeuronPosition(l, n);
              const isActive =
                active?.layer === l && active?.neuron === n;

              return (
                <circle
                  key={`neuron-${l}-${n}`}
                  cx={pos.x}
                  cy={pos.y}
                  r={neuronRadius}
                  fill={isActive ? "#f59e0b" : "#3b82f6"}
                  opacity={isActive ? 1 : 0.6}
                  className="transition-all"
                />
              );
            })
        )}

        {/* Layer labels */}
        {allLayers.map((layer, l) => (
          <text
            key={`label-${l}`}
            x={padding + l * layerSpacing}
            y={svgHeight - 10}
            textAnchor="middle"
            fontSize="11"
            fill="currentColor"
            opacity="0.6"
          >
            {l === 0
              ? "Input"
              : l === allLayers.length - 1
                ? "Output"
                : `Hidden ${l}`}
          </text>
        ))}

        {/* Unit counts */}
        {allLayers.map((layer, l) => (
          <text
            key={`count-${l}`}
            x={padding + l * layerSpacing}
            y={15}
            textAnchor="middle"
            fontSize="10"
            fill="currentColor"
            opacity="0.6"
          >
            {layer.units}
          </text>
        ))}
      </svg>

      {/* Layer info */}
      <div className="mt-4 text-xs text-foreground/70 space-y-1">
        <p>
          <strong>Layers:</strong> {allLayers.length} (Input → Hidden → Output)
        </p>
        <p>
          <strong>Total Neurons:</strong> {allLayers.reduce((sum, l) => sum + l.units, 0)}
        </p>
        <p>
          <strong>Activations:</strong>{" "}
          {layers.map((l) => l.activation).join(", ")}
        </p>
      </div>
    </div>
  );
}
