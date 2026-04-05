import { Vector } from "../types";

/**
 * Activation Functions for Neural Networks.
 * Used in forward pass of neurons.
 */

export const ActivationFunctions = {
  /**
   * Sigmoid: 1 / (1 + e^-x)
   * Maps input to (0, 1).
   */
  sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  },

  /**
   * Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
   */
  sigmoidDerivative(x: number): number {
    const s = this.sigmoid(x);
    return s * (1 - s);
  },

  /**
   * ReLU: max(0, x)
   * Returns 0 for negative values, x for positive.
   */
  relu(x: number): number {
    return Math.max(0, x);
  },

  /**
   * ReLU derivative: 0 if x < 0, 1 if x > 0
   */
  reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  },

  /**
   * Leaky ReLU: max(0.01x, x)
   */
  leakyRelu(x: number, alpha = 0.01): number {
    return x > 0 ? x : alpha * x;
  },

  /**
   * Leaky ReLU derivative
   */
  leakyReluDerivative(x: number, alpha = 0.01): number {
    return x > 0 ? 1 : alpha;
  },

  /**
   * Tanh: (e^x - e^-x) / (e^x + e^-x)
   * Maps input to (-1, 1).
   */
  tanh(x: number): number {
    return Math.tanh(x);
  },

  /**
   * Tanh derivative: 1 - tanh(x)^2
   */
  tanhDerivative(x: number): number {
    const t = Math.tanh(x);
    return 1 - t * t;
  },

  /**
   * Linear / Identity: f(x) = x
   */
  linear(x: number): number {
    return x;
  },

  /**
   * Linear derivative: 1
   */
  linearDerivative(): number {
    return 1;
  },

  /**
   * Softmax: e^x_i / sum(e^x_j)
   * Converts vector to probability distribution.
   */
  softmax(v: Vector): Vector {
    const maxVal = Math.max(...v);
    const exps = v.map((x) => Math.exp(x - maxVal)); // Subtract max for numerical stability
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  },

  /**
   * Element-wise sigmoid on a vector.
   */
  sigmoidVector(v: Vector): Vector {
    return v.map((x) => this.sigmoid(x));
  },

  /**
   * Element-wise ReLU on a vector.
   */
  reluVector(v: Vector): Vector {
    return v.map((x) => this.relu(x));
  },

  /**
   * Element-wise tanh on a vector.
   */
  tanhVector(v: Vector): Vector {
    return v.map((x) => this.tanh(x));
  },
};

/**
 * Loss Functions for training.
 * Measures how far predictions are from targets.
 */

export const LossFunctions = {
  /**
   * Mean Squared Error (MSE): (1/n) * sum((y - y_pred)^2)
   * For regression.
   */
  mse(targets: Vector, predictions: Vector): number {
    if (targets.length !== predictions.length) {
      throw new Error("Target and prediction dimension mismatch");
    }
    const squaredErrors = targets.map((t, i) => Math.pow(t - predictions[i], 2));
    return squaredErrors.reduce((a, b) => a + b, 0) / targets.length;
  },

  /**
   * Binary Cross-Entropy: -(1/n) * sum(y*log(y_pred) + (1-y)*log(1-y_pred))
   * For binary classification.
   */
  binaryCrossEntropy(targets: Vector, predictions: Vector): number {
    if (targets.length !== predictions.length) {
      throw new Error("Target and prediction dimension mismatch");
    }
    const epsilon = 1e-7; // Prevent log(0)
    const clipped = predictions.map((p) => Math.max(epsilon, Math.min(1 - epsilon, p)));

    const losses = targets.map((t, i) => {
      return -(t * Math.log(clipped[i]) + (1 - t) * Math.log(1 - clipped[i]));
    });
    return losses.reduce((a, b) => a + b, 0) / targets.length;
  },

  /**
   * Categorical Cross-Entropy: -sum(y * log(y_pred))
   * For multi-class classification.
   */
  categoricalCrossEntropy(targets: Vector, predictions: Vector): number {
    if (targets.length !== predictions.length) {
      throw new Error("Target and prediction dimension mismatch");
    }
    const epsilon = 1e-7;
    const clipped = predictions.map((p) => Math.max(epsilon, p));

    const losses = targets.map((t, i) => {
      return -t * Math.log(clipped[i]);
    });
    return losses.reduce((a, b) => a + b, 0);
  },

  /**
   * Mean Absolute Error (MAE): (1/n) * sum(|y - y_pred|)
   * For regression with less outlier sensitivity.
   */
  mae(targets: Vector, predictions: Vector): number {
    if (targets.length !== predictions.length) {
      throw new Error("Target and prediction dimension mismatch");
    }
    const absoluteErrors = targets.map((t, i) => Math.abs(t - predictions[i]));
    return absoluteErrors.reduce((a, b) => a + b, 0) / targets.length;
  },
};

/**
 * Regularization Functions.
 * Prevents overfitting by penalizing large weights.
 */

export const RegularizationFunctions = {
  /**
   * L1 Regularization (Lasso): lambda * sum(|w|)
   */
  l1(weights: Vector, lambda: number): number {
    return lambda * weights.reduce((acc, w) => acc + Math.abs(w), 0);
  },

  /**
   * L2 Regularization (Ridge): (lambda/2) * sum(w^2)
   */
  l2(weights: Vector, lambda: number): number {
    return (lambda / 2) * weights.reduce((acc, w) => acc + w * w, 0);
  },

  /**
   * Elastic Net: combines L1 and L2
   */
  elasticNet(weights: Vector, lambda: number, alpha: number): number {
    return (
      alpha * this.l1(weights, lambda) + (1 - alpha) * this.l2(weights, lambda)
    );
  },
};
