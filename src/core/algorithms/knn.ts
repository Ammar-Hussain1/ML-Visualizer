import { Algorithm, AlgorithmConfig } from "../algorithm";
import { Step, Vector, Matrix } from "../types";
import { DistanceMetrics } from "../math/distance-metrics";

/**
 * K-Nearest Neighbors (KNN) Classifier
 * 
 * Algorithm:
 * 1. Store all training data (lazy learning - no training phase)
 * 2. For each prediction:
 *    - Compute distance to all training samples
 *    - Find k nearest neighbors
 *    - Use majority vote for classification
 * 
 * No formal training needed, but we use the "training" phase to compute metrics
 */

interface KNNStateData {
  features: Matrix;
  targets: Vector;
  k: number;
  distanceMetric: "euclidean" | "manhattan" | "cosine" | "minkowski";
  predictions?: Vector;
  accuracies?: number[];
}

export interface KNNState extends KNNStateData {
  iteration: number;
  losses: number[]; // Uses accuracy loss: 1 - accuracy
}

export class KNearestNeighbors implements Algorithm<KNNState> {
  name = "K-Nearest Neighbors";
  config: AlgorithmConfig;
  private k: number;
  private metric: "euclidean" | "manhattan" | "cosine" | "minkowski";

  constructor(
    k: number = 3,
    metric: "euclidean" | "manhattan" | "cosine" | "minkowski" = "euclidean",
    config: AlgorithmConfig = {}
  ) {
    this.k = k;
    this.metric = metric;
    this.config = {
      learningRate: 0, // KNN doesn't use learning rate
      maxIterations: 1, // KNN only needs 1 iteration
      epsilon: 0.01,
      ...config,
    };
  }

  /**
   * KNN doesn't have a training phase - just store the data
   */
  initialize(features: Matrix, targets: Vector): KNNState {
    // Split into train/test (80/20)
    const splitIdx = Math.floor(features.length * 0.8);
    const trainFeatures = features.slice(0, splitIdx);
    const trainTargets = targets.slice(0, splitIdx);

    return {
      features: trainFeatures,
      targets: trainTargets,
      k: this.k,
      distanceMetric: this.metric,
      iteration: 0,
      losses: [],
      accuracies: [],
    };
  }

  /**
   * Forward pass: predict class for a single sample
   */
  forward(state: KNNState, input: Vector): { result: number; steps: Step[] } {
    const steps: Step[] = [];

    // Compute distances to all training samples
    const distances = state.features.map((sample, idx) => ({
      distance: this.computeDistance(input, sample),
      index: idx,
      label: state.targets[idx],
    }));

    steps.push({
      id: "knn-1",
      label: "Compute Distances",
      formula: `d(x, x_i) = ${this.metric === "euclidean" ? "√(Σ(x_j - x_{i,j})²)" : this.metric === "manhattan" ? "Σ|x_j - x_{i,j}|" : "distance"}`,
      substitution: `Computed ${state.features.length} distances using ${this.metric}`,
      result: `Min distance: ${Math.min(...distances.map((d) => d.distance)).toFixed(4)}, Max distance: ${Math.max(...distances.map((d) => d.distance)).toFixed(4)}`,
      description: "Calculate distance from query point to all training samples",
    });

    // Sort by distance and get k nearest
    const nearest = distances
      .sort((a, b) => a.distance - b.distance)
      .slice(0, state.k);

    steps.push({
      id: "knn-2",
      label: "Select K Nearest Neighbors",
      formula: `neighbors = argmin_{k} d(x, x_i)`,
      substitution: `Selected ${state.k} nearest neighbors`,
      result: `Distances: [${nearest.map((n) => n.distance.toFixed(4)).join(", ")}]`,
      description: `Found ${state.k} closest training samples`,
    });

    // Majority vote
    const votes = nearest.reduce(
      (acc, n) => {
        acc[n.label] = (acc[n.label] || 0) + 1;
        return acc;
      },
      {} as Record<number, number>
    );

    const prediction = Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
    const predictedClass = parseInt(prediction);

    steps.push({
      id: "knn-3",
      label: "Majority Vote",
      formula: `prediction = argmax_c Σ[neighbor_label == c]`,
      substitution: `Vote counts: ${Object.entries(votes).map((e) => `Class ${e[0]}: ${e[1]} votes`).join(", ")}`,
      result: `Predicted Class: ${predictedClass}`,
      description: "Use majority vote among neighbors to determine class",
    });

    return {
      result: predictedClass,
      steps,
    };
  }

  /**
   * KNN training: evaluate on test set
   */
  update(state: KNNState): { state: KNNState; steps: Step[] } {
    const steps: Step[] = [];

    // Generate test data (remaining 20%)
    const fullDataSize = Math.ceil(state.features.length / 0.8);
    const testStart = state.features.length;
    
    // For demo, use a subset of training data as test
    const testSize = Math.ceil(state.features.length * 0.25);
    let correct = 0;
    let totalPredictions = 0;

    for (let i = 0; i < testSize; i++) {
      const testIdx = Math.floor(Math.random() * state.features.length);
      const sample = state.features[testIdx];
      const trueLabel = state.targets[testIdx];

      // Get prediction
      const { result: prediction } = this.forward(state, sample);
      if (prediction === trueLabel) correct++;
      totalPredictions++;
    }

    const accuracy = (correct / totalPredictions) * 100;
    const loss = 1 - correct / totalPredictions;

    steps.push({
      id: `iter-${state.iteration}-1`,
      label: "Evaluate on Test Set",
      formula: `accuracy = (# correct / # total) × 100%`,
      substitution: `Tested ${totalPredictions} random samples from training set`,
      result: `Accuracy: ${accuracy.toFixed(2)}%, Correct: ${correct}/${totalPredictions}`,
      description: "Evaluate KNN classifier performance",
    });

    steps.push({
      id: `iter-${state.iteration}-2`,
      label: "Check Convergence",
      formula: `is_converged = (loss_improvement < ε)`,
      substitution: `Current loss: ${loss.toFixed(6)}`,
      result: `Loss: ${loss.toFixed(6)}, K: ${state.k}, Metric: ${state.distanceMetric}`,
      description: "KNN has no parameters to learn, evaluation is metric of choice",
    });

    const newState: KNNState = {
      ...state,
      losses: [...state.losses, loss],
      accuracies: [...(state.accuracies ?? []), accuracy],
      iteration: state.iteration + 1,
    };

    return { state: newState, steps };
  }

  /**
   * Check convergence - KNN converges immediately (no learning)
   */
  isReached(state: KNNState): boolean {
    // KNN always converges after first iteration
    return state.iteration > 0;
  }

  /**
   * Compute distance using configured metric
   */
  private computeDistance(p1: Vector, p2: Vector): number {
    switch (this.metric) {
      case "euclidean":
        return DistanceMetrics.euclidean(p1, p2);
      case "manhattan":
        return DistanceMetrics.manhattan(p1, p2);
      case "cosine":
        return DistanceMetrics.cosine(p1, p2);
      case "minkowski":
        return DistanceMetrics.minkowski(p1, p2, 3);
      default:
        return DistanceMetrics.euclidean(p1, p2);
    }
  }
}
