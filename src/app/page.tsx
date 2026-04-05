"use client";

import { useEffect, useState } from "react";
import { AlgorithmLayout } from "@/components/AlgorithmLayout";
import { ControlPanel } from "@/components/ControlPanel";
import { VisualizationPanel } from "@/components/VisualizationPanel";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { NeuralNetworkVisualization } from "@/components/NeuralNetworkVisualization";
import { KNNVisualization } from "@/components/KNNVisualization";
import { DecisionTreeVisualization } from "@/components/DecisionTreeVisualization";
import { useVisualizerStore } from "@/core/store";
import { LinearRegression } from "@/core/algorithms/linear-regression";
import { LogisticRegression } from "@/core/algorithms/logistic-regression";
import { NeuralNetwork } from "@/core/algorithms/neural-network";
import { KNearestNeighbors } from "@/core/algorithms/knn";
import { DecisionTree } from "@/core/algorithms/decision-tree";
import { TrainingEngine } from "@/core/training-engine";
import { VisualizationEngine } from "@/core/visualization-engine";
import { ChartConfig } from "@/core/visualization-engine";
import { Step } from "@/core/types";
import { Matrix, Vector } from "@/core/types";

// Generate synthetic training data for classification
function generateClassificationData(
  samples: number = 100,
  numClasses: number = 3
): { features: Matrix; targets: Vector } {
  const features: Matrix = [];
  const targets: Vector = [];

  for (let i = 0; i < samples; i++) {
    const angle = (Math.random() * 2 * Math.PI);
    const radius = Math.random() * 5;
    const classIdx = Math.floor(Math.random() * numClasses);

    // Create class-specific offset
    const classAngle = (classIdx * 2 * Math.PI) / numClasses;
    const classRadius = 2;

    const x =
      classRadius * Math.cos(classAngle) +
      radius * Math.cos(angle) +
      (Math.random() - 0.5) * 0.5;
    const y =
      classRadius * Math.sin(classAngle) +
      radius * Math.sin(angle) +
      (Math.random() - 0.5) * 0.5;

    features.push([x, y]);
    targets.push(classIdx);
  }

  return { features, targets };
}

export default function Home() {
  const store = useVisualizerStore();
  const [chart, setChart] = useState<ChartConfig | undefined>();
  const [currentVisualization, setCurrentVisualization] = useState<"chart" | "nn" | "knn" | "dt">("chart");
  const [trainData, setTrainData] = useState<{ features: Matrix; targets: Vector } | undefined>();
  const [isTraining, setIsTraining] = useState(false);

  const handleTrain = async (config: Record<string, unknown>) => {
    setIsTraining(true);

    // Reset store
    store.resetVisualization();

    try {
      const algorithmName = config.algorithm as string;
      let algorithm: any;
      let features: Matrix = [];
      let targets: Vector = [];

      if (algorithmName === "Linear Regression") {
        // Generate regression data
        features = [];
        targets = [];
        for (let i = 0; i < (config.dataSize as number) || 50; i++) {
          const x = Math.random() * 10 - 5;
          const noise = (Math.random() - 0.5) * 1;
          const y = 2 * x + 1 + noise;
          features.push([x]);
          targets.push(y);
        }
        algorithm = new LinearRegression({
          learningRate: (config.learningRate as number) || 0.01,
          maxIterations: (config.maxIterations as number) || 100,
        });
        setTrainData({ features, targets });
        setCurrentVisualization("chart");
      } else if (algorithmName === "Logistic Regression") {
        // Generate classification data (binary)
        const data = generateClassificationData(
          (config.dataSize as number) || 50,
          2
        );
        features = data.features;
        targets = data.targets;
        algorithm = new LogisticRegression({
          learningRate: (config.learningRate as number) || 0.01,
          maxIterations: (config.maxIterations as number) || 100,
        });
        setTrainData({ features, targets });
        setCurrentVisualization("chart");
      } else if (algorithmName === "KNN") {
        // Generate multi-class classification data
        const data = generateClassificationData(
          (config.dataSize as number) || 100,
          3
        );
        features = data.features;
        targets = data.targets;

        algorithm = new KNearestNeighbors(
          5, // k=5
          "euclidean",
          {
            learningRate: 0,
            maxIterations: 1,
          }
        );
        setTrainData({ features, targets });
        setCurrentVisualization("knn");
      } else if (algorithmName === "Decision Tree") {
        // Generate multi-class classification data
        const data = generateClassificationData(
          (config.dataSize as number) || 100,
          3
        );
        features = data.features;
        targets = data.targets;

        algorithm = new DecisionTree(
          5, // max depth
          2, // min samples split
          {
            learningRate: 0,
            maxIterations: 1,
          }
        );
        setTrainData({ features, targets });
        setCurrentVisualization("dt");
      } else if (algorithmName === "Neural Network") {
        // Generate multi-class classification data
        const data = generateClassificationData(
          (config.dataSize as number) || 100,
          3
        );
        features = data.features;
        targets = data.targets;

        algorithm = new NeuralNetwork(
          [
            { units: 8, activation: "relu" },
            { units: 4, activation: "relu" },
            { units: 3, activation: "softmax" },
          ],
          {
            learningRate: (config.learningRate as number) || 0.01,
            maxIterations: (config.maxIterations as number) || 100,
          }
        );
        setTrainData({ features, targets });
        setCurrentVisualization("nn");
      }

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
            algorithms={["Linear Regression", "Logistic Regression", "Neural Network", "KNN", "Decision Tree"]}
            onTrain={handleTrain}
          />
        }
        visualization={
          currentVisualization === "nn" ? (
            <NeuralNetworkVisualization
              layers={[
                { units: 8, activation: "relu" },
                { units: 4, activation: "relu" },
                { units: 3, activation: "softmax" },
              ]}
            />
          ) : currentVisualization === "knn" ? (
            <KNNVisualization trainData={trainData} k={5} metric="euclidean" />
          ) : currentVisualization === "dt" ? (
            <DecisionTreeVisualization
              algorithm={store.currentAlgorithm as unknown as DecisionTree}
              trainData={trainData}
            />
          ) : (
            <VisualizationPanel chart={chart} isLoading={isTraining} />
          )
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
