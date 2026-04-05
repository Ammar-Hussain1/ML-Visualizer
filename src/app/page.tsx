"use client";

import { useEffect, useState } from "react";
import { AlgorithmLayout } from "@/components/AlgorithmLayout";
import { ControlPanel } from "@/components/ControlPanel";
import { VisualizationPanel } from "@/components/VisualizationPanel";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { useVisualizerStore } from "@/core/store";
import { LinearRegression } from "@/core/algorithms/linear-regression";
import { TrainingEngine } from "@/core/training-engine";
import { VisualizationEngine } from "@/core/visualization-engine";
import { ChartConfig } from "@/core/visualization-engine";
import { Step } from "@/core/types";
import { Matrix, Vector } from "@/core/types";

// Generate synthetic training data
function generateTrainingData(
  samples: number = 50
): { features: Matrix; targets: Vector } {
  const features: Matrix = [];
  const targets: Vector = [];

  for (let i = 0; i < samples; i++) {
    const x = Math.random() * 10 - 5;
    const noise = (Math.random() - 0.5) * 1;
    const y = 2 * x + 1 + noise;

    features.push([x]);
    targets.push(y);
  }

  return { features, targets };
}

export default function Home() {
  const store = useVisualizerStore();
  const [chart, setChart] = useState<ChartConfig | undefined>();
  const [isTraining, setIsTraining] = useState(false);

  const handleTrain = async (config: Record<string, unknown>) => {
    setIsTraining(true);

    // Reset store
    store.resetVisualization();

    try {
      const { features, targets } = generateTrainingData(
        (config.dataSize as number) || 50
      );
      const algorithm = new LinearRegression({
        learningRate: (config.learningRate as number) || 0.01,
        maxIterations: (config.maxIterations as number) || 100,
      });

      const engine = new TrainingEngine(algorithm, {
        onStepComplete: (step, progress) => {
          store.addStep(step);
          store.setCurrentStepIndex(progress.steps.length - 1);
        },
        onIterationComplete: (progress) => {
          // Update visualization
          const visEngine = new VisualizationEngine();
          const state = progress.currentState.algorithmState as any;

          if (state?.losses?.length > 0) {
            const lossChart = visEngine.lossCurve(
              state.losses,
              "Training Loss"
            );
            setChart(lossChart);
          }
        },
      });

      engine.initialize(features, targets);
      await engine.train((config.maxIterations as number) || 100);

      setIsTraining(false);
    } catch (error) {
      console.error("Training error:", error);
      store.setError((error as Error).message);
      setIsTraining(false);
    }
  };

  return (
    <main className="w-full h-screen overflow-hidden">
      <AlgorithmLayout
        controlPanel={
          <ControlPanel
            algorithms={["Linear Regression", "Logistic Regression"]}
            onTrain={handleTrain}
          />
        }
        visualization={
          <VisualizationPanel chart={chart} isLoading={isTraining} />
        }
        explanation={
          <ExplanationPanel
            steps={store.steps}
            currentStepIndex={store.currentStepIndex}
          />
        }
      />
    </main>
  );
}
