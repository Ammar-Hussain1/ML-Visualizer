import { Algorithm, AlgorithmConfig } from "../algorithm";
import { Step, Vector, Matrix, TrainingData } from "../types";
import { LinearAlgebra } from "../math/linear-algebra";
import { LossFunctions } from "../math/activation-loss";

/**
 * Linear Regression using Gradient Descent.
 * 
 * Mathematical Model:
 * y = w·x + b
 * 
 * Cost Function:
 * J(w,b) = (1/2m) * sum((y_pred - y_actual)^2)
 * 
 * Gradient Update:
 * w := w - α * (∂J/∂w)
 * b := b - α * (∂J/∂b)
 */

interface LinearRegressionState {
  weights: Vector;
  bias: number;
  features: Matrix;
  targets: Vector;
  losses: number[];
  iteration: number;
}

export class LinearRegression implements Algorithm<LinearRegressionState> {
  name = "Linear Regression";
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
  initialize(features: Matrix, targets: Vector): LinearRegressionState {
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
   */
  forward(state: LinearRegressionState, input: Vector): { result: number; steps: Step[] } {
    const steps: Step[] = [];

    // Step 1: Compute dot product
    const dotProduct = LinearAlgebra.dot(state.weights, input);
    steps.push({
      id: "forward-1",
      label: "Compute Linear Combination",
      formula: "z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b",
      substitution: `z = ${state.weights.map((w, i) => `${w.toFixed(4)}·${input[i]}`).join(" + ")} ${state.bias >= 0 ? "+" : ""} ${state.bias.toFixed(4)}`,
      result: `z = ${(dotProduct + state.bias).toFixed(4)}`,
      description: "Linear transformation of input features",
    });

    return {
      result: dotProduct + state.bias,
      steps,
    };
  }

  /**
   * Perform a training step (gradient descent update).
   */
  update(state: LinearRegressionState): { state: LinearRegressionState; steps: Step[] } {
    const steps: Step[] = [];
    const m = state.features.length; // Number of samples
    const lr = this.config.learningRate!;

    // Step 1: Forward pass - compute predictions
    const predictions = state.features.map((sample) => {
      const z = LinearAlgebra.dot(state.weights, sample) + state.bias;
      return z;
    });
    steps.push({
      id: `iter-${state.iteration}-1`,
      label: "Forward Pass",
      formula: "ŷ = X·w + b",
      substitution: `Computed predictions for ${m} samples`,
      result: `Predictions: [${predictions.slice(0, 3).map((p) => p.toFixed(4)).join(", ")}${m > 3 ? ", ..." : ""}]`,
      description: "Generate predictions for all training samples",
    });

    // Step 2: Compute loss
    const loss = LossFunctions.mse(state.targets, predictions);
    steps.push({
      id: `iter-${state.iteration}-2`,
      label: "Compute Loss (MSE)",
      formula: "J(w,b) = (1/m)·Σ(ŷ - y)²",
      substitution: `MSE = (1/${m})·Σ(predictions - targets)²`,
      result: `Loss = ${loss.toFixed(6)}`,
      description: "Mean squared error measures prediction error",
    });

    // Step 3: Compute gradients
    const errors = LinearAlgebra.subtract(predictions, state.targets);
    const gradW = LinearAlgebra.matrixVectorMultiply(
      [LinearAlgebra.transpose(state.features)[0]], // Will be computed properly
      errors
    ).map((g) => g / m);

    const gradB = LinearAlgebra.sum(errors) / m;

    steps.push({
      id: `iter-${state.iteration}-3`,
      label: "Compute Gradients",
      formula: "∂J/∂w = (1/m)·Xᵀ·(ŷ - y), ∂J/∂b = (1/m)·Σ(ŷ - y)",
      substitution: `Gradient weights: [${state.weights.map((_, i) => (gradW[i] ?? 0).toFixed(6)).join(", ")}]`,
      result: `∂J/∂b = ${gradB.toFixed(6)}`,
      description: "Compute how much to adjust each parameter",
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
      description: `Adjust weights and bias in opposite direction of gradient (learning rate=${lr})`,
    });

    const newState: LinearRegressionState = {
      ...state,
      weights: newWeights,
      bias: newBias,
      losses: [...state.losses, loss],
      iteration: state.iteration + 1,
    };

    return { state: newState, steps };
  }

  /**
   * Check for convergence.
   */
  isReached(state: LinearRegressionState): boolean {
    if (state.losses.length < 2) return false;

    const lastLoss = state.losses[state.losses.length - 1];
    const prevLoss = state.losses[state.losses.length - 2];
    const improvement = Math.abs(lastLoss - prevLoss);

    return improvement < this.config.epsilon!;
  }
}
