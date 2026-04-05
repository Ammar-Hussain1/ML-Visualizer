import { Algorithm, AlgorithmConfig } from "../algorithm";
import { Step, Vector, Matrix } from "../types";
import { LinearAlgebra } from "../math/linear-algebra";
import { LossFunctions, ActivationFunctions } from "../math/activation-loss";

/**
 * Logistic Regression using Gradient Descent.
 * 
 * Mathematical Model (Binary Classification):
 * σ(z) = 1 / (1 + e^-z)    [Sigmoid function]
 * ŷ = σ(w·x + b)
 * 
 * Cost Function (Binary Cross-Entropy):
 * J(w,b) = -(1/m) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
 * 
 * Gradient Update:
 * w := w - α * (∂J/∂w)
 * b := b - α * (∂J/∂b)
 * 
 * Decision Boundary:
 * if ŷ ≥ 0.5 → predict class 1
 * if ŷ < 0.5 → predict class 0
 */

interface LogisticRegressionState {
  weights: Vector;
  bias: number;
  features: Matrix;
  targets: Vector;
  losses: number[];
  iteration: number;
  predictions?: Vector;
  accuracies?: number[];
}

export class LogisticRegression
  implements Algorithm<LogisticRegressionState>
{
  name = "Logistic Regression";
  config: AlgorithmConfig;

  constructor(config: AlgorithmConfig = {}) {
    this.config = {
      learningRate: 0.01,
      maxIterations: 100,
      epsilon: 1e-6,
      ...config,
    };
  }

  /**
   * Initialize weights and bias.
   */
  initialize(features: Matrix, targets: Vector): LogisticRegressionState {
    const n = features[0].length; // Number of features
    const weights = LinearAlgebra.randomGaussian(n, 0, 0.01);

    return {
      weights,
      bias: 0,
      features,
      targets,
      losses: [],
      iteration: 0,
    };
  }

  /**
   * Make a prediction for a single sample.
   * Returns probability and corresponding class.
   */
  forward(
    state: LogisticRegressionState,
    input: Vector
  ): {
    result: { probability: number; class: number };
    steps: Step[];
  } {
    const steps: Step[] = [];

    // Step 1: Compute linear combination
    const z = LinearAlgebra.dot(state.weights, input) + state.bias;
    steps.push({
      id: "forward-1",
      label: "Compute Linear Combination",
      formula: "z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b",
      substitution: `z = ${state.weights.map((w, i) => `${w.toFixed(4)}·${input[i]}`).join(" + ")} ${state.bias >= 0 ? "+" : ""} ${state.bias.toFixed(4)}`,
      result: `z = ${z.toFixed(4)}`,
      description: "Linear transformation of input features",
    });

    // Step 2: Apply sigmoid
    const probability = ActivationFunctions.sigmoid(z);
    steps.push({
      id: "forward-2",
      label: "Apply Sigmoid Activation",
      formula: "σ(z) = 1 / (1 + e^-z)",
      substitution: `σ(${z.toFixed(4)}) = 1 / (1 + e^-${z.toFixed(4)})`,
      result: `ŷ = ${probability.toFixed(4)} (probability of class 1)`,
      description: "Convert linear output to probability in [0, 1]",
    });

    // Step 3: Determine class
    const predictedClass = probability >= 0.5 ? 1 : 0;
    steps.push({
      id: "forward-3",
      label: "Make Classification Decision",
      formula: "predicted_class = 1 if σ(z) ≥ 0.5 else 0",
      substitution: `${probability.toFixed(4)} ${probability >= 0.5 ? "≥" : "<"} 0.5`,
      result: `Predicted Class = ${predictedClass}`,
      description: "Threshold at 0.5 to make binary classification",
    });

    return {
      result: { probability, class: predictedClass },
      steps,
    };
  }

  /**
   * Perform a training step (gradient descent update).
   */
  update(state: LogisticRegressionState): {
    state: LogisticRegressionState;
    steps: Step[];
  } {
    const steps: Step[] = [];
    const m = state.features.length;
    const lr = this.config.learningRate!;

    // Step 1: Forward pass - compute predicted probabilities
    const probabilities = state.features.map(
      (sample) =>
        ActivationFunctions.sigmoid(LinearAlgebra.dot(state.weights, sample) + state.bias)
    );

    steps.push({
      id: `iter-${state.iteration}-1`,
      label: "Forward Pass (Sigmoid)",
      formula: "ŷ = σ(X·w + b) for all samples",
      substitution: `Computed sigmoid probabilities for ${m} samples`,
      result: `Predictions: [${probabilities
        .slice(0, 3)
        .map((p) => p.toFixed(4))
        .join(", ")}${m > 3 ? ", ..." : ""}]`,
      description: "Generate probability predictions for all training samples",
    });

    // Step 2: Compute loss (binary cross-entropy)
    const loss = LossFunctions.binaryCrossEntropy(state.targets, probabilities);
    steps.push({
      id: `iter-${state.iteration}-2`,
      label: "Compute Loss (Binary Cross-Entropy)",
      formula:
        "J(w,b) = -(1/m)·Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]",
      substitution: `BCE = -(1/${m})·Σ[targets·log(predictions) + (1-targets)·log(1-predictions)]`,
      result: `Loss = ${loss.toFixed(6)}`,
      description: "Binary cross-entropy measures classification error",
    });

    // Step 3: Compute gradients
    const errors = LinearAlgebra.subtract(probabilities, state.targets);

    // Gradient for weights: (1/m) * X^T * (ŷ - y)
    const XTransposed = LinearAlgebra.transpose(state.features);
    const gradW = LinearAlgebra.scale(
      LinearAlgebra.matrixVectorMultiply([...XTransposed], errors),
      1 / m
    );

    // Gradient for bias: (1/m) * sum(ŷ - y)
    const gradB = LinearAlgebra.sum(errors) / m;

    steps.push({
      id: `iter-${state.iteration}-3`,
      label: "Compute Gradients",
      formula:
        "∂J/∂w = (1/m)·Xᵀ·(ŷ - y), ∂J/∂b = (1/m)·Σ(ŷ - y)",
      substitution: `Gradient weights: [${state.weights
        .map((_, i) => (gradW[i] ?? 0).toFixed(6))
        .join(", ")}]`,
      result: `∂J/∂b = ${gradB.toFixed(6)}`,
      description: "Compute how much each parameter contributed to the error",
    });

    // Step 4: Update parameters
    const newWeights = LinearAlgebra.subtract(
      state.weights,
      LinearAlgebra.scale(gradW, lr)
    );
    const newBias = state.bias - lr * gradB;

    steps.push({
      id: `iter-${state.iteration}-4`,
      label: "Update Parameters",
      formula: "w := w - α·∂J/∂w, b := b - α·∂J/∂b",
      substitution: `w := w - ${lr}·∂J/∂w`,
      result: `Updated bias: ${newBias.toFixed(6)}`,
      description: `Adjust parameters in opposite direction of gradient (learning rate=${lr})`,
    });

    // Step 5: Compute training accuracy
    const predictions = probabilities.map((p) => (p >= 0.5 ? 1 : 0));
    const correctPredictions = predictions.filter(
      (p, i) => p === state.targets[i]
    ).length;
    const accuracy = (correctPredictions / m) * 100;

    steps.push({
      id: `iter-${state.iteration}-5`,
      label: "Compute Training Accuracy",
      formula: "Accuracy = (# correct predictions / m) × 100%",
      substitution: `Accuracy = (${correctPredictions} / ${m}) × 100%`,
      result: `Training Accuracy: ${accuracy.toFixed(2)}%`,
      description: "Percentage of correct classifications",
    });

    const newState: LogisticRegressionState = {
      ...state,
      weights: newWeights,
      bias: newBias,
      losses: [...state.losses, loss],
      predictions: probabilities,
      accuracies: [...(state.accuracies ?? []), accuracy],
      iteration: state.iteration + 1,
    };

    return { state: newState, steps };
  }

  /**
   * Check for convergence.
   */
  isReached(state: LogisticRegressionState): boolean {
    if (state.losses.length < 2) return false;

    const lastLoss = state.losses[state.losses.length - 1];
    const prevLoss = state.losses[state.losses.length - 2];
    const improvement = Math.abs(lastLoss - prevLoss);

    return improvement < this.config.epsilon!;
  }
}
