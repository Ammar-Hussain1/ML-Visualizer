"use client";

import React, { useState } from "react";
import { useVisualizerStore } from "@/core/store";

/**
 * Control Panel: User inputs and hyperparameters
 */

interface ControlPanelProps {
  algorithms: string[];
  onTrain?: (config: Record<string, unknown>) => void;
}

export function ControlPanel({ algorithms, onTrain }: ControlPanelProps) {
  const { setConfig, setCurrentAlgorithm } = useVisualizerStore();

  const [selectedAlgorithm, setSelectedAlgorithm] = useState(algorithms[0] || "");
  const [learningRate, setLearningRate] = useState(0.01);
  const [maxIterations, setMaxIterations] = useState(100);
  const [epsilon, setEpsilon] = useState(1e-6);
  const [dataSize, setDataSize] = useState(100);

  const handleTrainClick = () => {
    const config = {
      algorithm: selectedAlgorithm,
      learningRate,
      maxIterations,
      epsilon,
      dataSize,
    };
    setConfig(config);
    setCurrentAlgorithm(selectedAlgorithm);
    onTrain?.(config);
  };

  return (
    <div className="space-y-6">
      {/* Algorithm Selection */}
      <div>
        <label className="block text-sm font-semibold mb-2">Algorithm</label>
        <select
          value={selectedAlgorithm}
          onChange={(e) => setSelectedAlgorithm(e.target.value)}
          className="w-full px-3 py-2 bg-background border border-border rounded-md text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
        >
          {algorithms.map((algo) => (
            <option key={algo} value={algo}>
              {algo}
            </option>
          ))}
        </select>
      </div>

      {/* Hyperparameters */}
      <div className="pt-4 border-t border-border">
        <h3 className="font-semibold mb-4">Hyperparameters</h3>

        {/* Learning Rate */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Learning Rate: {learningRate.toFixed(6)}
          </label>
          <input
            type="range"
            min="0.0001"
            max="1"
            step="0.0001"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Max Iterations */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Max Iterations: {maxIterations}
          </label>
          <input
            type="range"
            min="10"
            max="1000"
            step="10"
            value={maxIterations}
            onChange={(e) => setMaxIterations(parseInt(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Epsilon (Convergence) */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Convergence Threshold (ε): {epsilon.toExponential(2)}
          </label>
          <input
            type="range"
            min="-10"
            max="-2"
            step="1"
            value={Math.log10(epsilon)}
            onChange={(e) => setEpsilon(Math.pow(10, parseFloat(e.target.value)))}
            className="w-full"
          />
        </div>

        {/* Data Size */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Data Size: {dataSize}
          </label>
          <input
            type="range"
            min="10"
            max="1000"
            step="10"
            value={dataSize}
            onChange={(e) => setDataSize(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* Action Buttons */}
      <div className="pt-4 border-t border-border space-y-2">
        <button
          onClick={handleTrainClick}
          className="w-full px-4 py-2 bg-primary text-primary-foreground font-semibold rounded-md hover:bg-primary/90 transition-colors"
        >
          Start Training
        </button>
        <button className="w-full px-4 py-2 bg-secondary text-secondary-foreground font-semibold rounded-md hover:bg-secondary/90 transition-colors">
          Pause
        </button>
        <button className="w-full px-4 py-2 bg-muted text-muted-foreground font-semibold rounded-md hover:bg-muted/90 transition-colors">
          Reset
        </button>
      </div>

      {/* Info Box */}
      <div className="p-3 bg-card rounded-md border border-border text-sm">
        <p className="font-semibold mb-2">💡 How it works:</p>
        <ul className="text-xs space-y-1 opacity-80">
          <li>• Adjust hyperparameters above</li>
          <li>• Click "Start Training" to begin</li>
          <li>• Watch the visualization update</li>
          <li>• See mathematical steps on the right</li>
        </ul>
      </div>
    </div>
  );
}
