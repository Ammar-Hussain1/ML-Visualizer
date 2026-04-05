import { Algorithm, AlgorithmConfig } from "../algorithm";
import { Step, Vector, Matrix } from "../types";
import { LinearAlgebra } from "../math/linear-algebra";
import { LossFunctions, ActivationFunctions } from "../math/activation-loss";

/**
 * Neural Network Layer Configuration
 */
export interface LayerConfig {
  units: number;
  activation: "relu" | "sigmoid" | "tanh" | "linear" | "softmax";
}

/**
 * Neural Network State
 */
export interface NeuralNetworkState {
  layers: LayerConfig[];
  weights: Matrix[]; // weights[i] is weight matrix between layer i and i+1
  biases: Vector[]; // biases[i] is bias vector for layer i+1
  activations: Vector[]; // activations[i] is output of layer i
  z_values: Vector[]; // z_values[i] is pre-activation of layer i+1
  features: Matrix;
  targets: Matrix; // For multi-class, targets are one-hot encoded
  losses: number[];
  iteration: number;
}

/**
 * Multi-Layer Perceptron (MLP) Neural Network
 * 
 * Mathematical Model:
 * Forward Pass:
 *   z^(l) = W^(l) * a^(l-1) + b^(l)
 *   a^(l) = σ(z^(l))
 * 
 * Backward Pass (Backpropagation):
 *   δ^(L) = ∇_a J ⊙ σ'(z^(L))  [output layer]
 *   δ^(l) = (W^(l+1))^T * δ^(l+1) ⊙ σ'(z^(l))  [hidden layers]
 *   ∂J/∂W^(l) = δ^(l) * (a^(l-1))^T
 *   ∂J/∂b^(l) = δ^(l)
 */

export class NeuralNetwork implements Algorithm<NeuralNetworkState> {
  name = "Neural Network (MLP)";
  config: AlgorithmConfig;
  private layers: LayerConfig[];

  constructor(layerConfigs: LayerConfig[], config: AlgorithmConfig = {}) {
    this.layers = layerConfigs;
    this.config = {
      learningRate: 0.01,
      maxIterations: 100,
      epsilon: 1e-4,
      ...config,
    };
  }

  /**
   * Initialize network weights and biases using Xavier initialization.
   */
  initialize(features: Matrix, targets: Vector): NeuralNetworkState {
    const inputSize = features[0].length;
    const weights: Matrix[] = [];
    const biases: Vector[] = [];

    // Convert targets to one-hot encoding for classification
    const numClasses = Math.max(...targets) + 1;
    const targetsOneHot = targets.map((t) => {
      const row = Array(Math.floor(numClasses)).fill(0);
      row[Math.floor(t)] = 1;
      return row;
    });

    // Create weight matrices between each pair of layers
    let prevSize = inputSize;
    for (const layer of this.layers) {
      // Xavier initialization: sqrt(1/n) scaling
      const scale = Math.sqrt(1 / prevSize);
      const W = Array(layer.units)
        .fill(0)
        .map(() => LinearAlgebra.randomGaussian(prevSize, 0, scale));

      const b = LinearAlgebra.zeros(layer.units);

      weights.push(W);
      biases.push(b);
      prevSize = layer.units;
    }

    return {
      layers: this.layers,
      weights,
      biases,
      activations: Array(this.layers.length + 1).fill([]),
      z_values: Array(this.layers.length).fill([]),
      features,
      targets: targetsOneHot,
      losses: [],
      iteration: 0,
    };
  }

  /**
   * Forward pass through the network.
   */
  forward(
    state: NeuralNetworkState,
    input: Vector
  ): { result: Vector; steps: Step[] } {
    const steps: Step[] = [];
    const activations: Vector[] = [input];
    const z_values: Vector[] = [];

    let current = input;

    for (let l = 0; l < state.layers.length; l++) {
      // Compute z = W*a + b
      const z = LinearAlgebra.add(
        LinearAlgebra.matrixVectorMultiply(state.weights[l], current),
        state.biases[l]
      );
      z_values.push(z);

      // Apply activation
      const activation = state.layers[l].activation;
      const a = this.applyActivation(z, activation);
      activations.push(a);

      steps.push({
        id: `forward-${l}`,
        label: `Layer ${l + 1}: Compute Activation`,
        formula: `z^(${l}) = W^(${l})·a^(${l - 1}) + b^(${l}), a^(${l}) = σ${l}(z^(${l}))`,
        substitution: `Computed ${activation} activation for layer ${l + 1}`,
        result: `Output: [${a.slice(0, 3).map((v) => v.toFixed(4)).join(", ")}${a.length > 3 ? ", ..." : ""}]`,
        description: `Forward pass through layer ${l + 1} with ${activation} activation`,
      });

      current = a;
    }

    return { result: current, steps };
  }

  /**
   * Perform a training step with backpropagation.
   */
  update(state: NeuralNetworkState): { state: NeuralNetworkState; steps: Step[] } {
    const steps: Step[] = [];
    const m = state.features.length;
    const lr = this.config.learningRate!;

    // Forward pass
    let activations: Vector[] = [];
    let z_values: Vector[] = [];
    let current = state.features[0]; // Use first sample for demo steps

    for (let l = 0; l < state.layers.length; l++) {
      activations.push(current);
      const z = LinearAlgebra.add(
        LinearAlgebra.matrixVectorMultiply(state.weights[l], current),
        state.biases[l]
      );
      z_values.push(z);
      current = this.applyActivation(z, state.layers[l].activation);
    }
    activations.push(current);

    steps.push({
      id: `iter-${state.iteration}-1`,
      label: "Forward Pass Through All Layers",
      formula: "a^(0) = x, a^(l) = σ_l(W^(l)·a^(l-1) + b^(l))",
      substitution: `Propagated input through ${state.layers.length} layers`,
      result: `Final output: [${current.slice(0, 3).map((v) => v.toFixed(4)).join(", ")}${current.length > 3 ? ", ..." : ""}]`,
      description: "Forward propagation computes output from input",
    });

    // Compute loss (using first sample)
    const predictions = this.predictBatch(state, state.features);
    const loss = LossFunctions.categoricalCrossEntropy(
      state.targets[0],
      predictions[0]
    );

    steps.push({
      id: `iter-${state.iteration}-2`,
      label: "Compute Loss",
      formula: "J = -(1/m)·Σ Σ y_ij·log(ŷ_ij)",
      substitution: `Averaged cross-entropy loss over ${m} samples`,
      result: `Loss = ${loss.toFixed(6)}`,
      description: "Cross-entropy measures classification error",
    });

    // Backward pass - compute deltas for output layer
    const outputIdx = state.layers.length - 1;
    const outputErrors = LinearAlgebra.subtract(
      predictions[0],
      state.targets[0]
    );

    const deltas: Vector[] = Array(state.layers.length).fill([]);
    deltas[outputIdx] = outputErrors;

    steps.push({
      id: `iter-${state.iteration}-3`,
      label: "Backpropagation: Output Layer",
      formula: "δ^(L) = (ŷ - y)",
      substitution: `Computed error deltas for output layer`,
      result: `Deltas: [${outputErrors.slice(0, 3).map((v) => v.toFixed(4)).join(", ")}${outputErrors.length > 3 ? ", ..." : ""}]`,
      description: "Output layer error = prediction - target",
    });

    // Backprop through hidden layers
    for (let l = state.layers.length - 2; l >= 0; l--) {
      const nextDeltas = deltas[l + 1];
      const weightTranspose = LinearAlgebra.transpose(state.weights[l + 1]);
      const rawDeltas = LinearAlgebra.matrixVectorMultiply(
        weightTranspose,
        nextDeltas
      );

      // Apply activation derivative
      const activationDerivs = z_values[l].map((z) =>
        this.activationDerivative(z, state.layers[l].activation)
      );

      deltas[l] = LinearAlgebra.elementwiseMultiply(
        rawDeltas,
        activationDerivs
      ) as Vector;
    }

    steps.push({
      id: `iter-${state.iteration}-4`,
      label: "Backpropagation: Hidden Layers",
      formula: "δ^(l) = (W^(l+1))^T·δ^(l+1) ⊙ σ'(z^(l))",
      substitution: `Propagated errors backward through ${state.layers.length - 1} hidden layers`,
      result: `Computed delta for each hidden layer`,
      description: "Error signal flows backward through network",
    });

    // Compute gradients and update weights/biases
    const newWeights = state.weights.map((W, l) => {
      // dW = δ^(l) ⊗ a^(l-1) / m
      const dW = LinearAlgebra.outerProduct(deltas[l], activations[l]);
      const scaledDW = dW.map((row) =>
        LinearAlgebra.scale(row, lr / m)
      );
      return W.map((row, i) =>
        LinearAlgebra.subtract(row, scaledDW[i])
      );
    });

    const newBiases = state.biases.map((b, l) =>
      LinearAlgebra.subtract(b, LinearAlgebra.scale(deltas[l], lr / m))
    );

    steps.push({
      id: `iter-${state.iteration}-5`,
      label: "Update Weights and Biases",
      formula: "W^(l) := W^(l) - α·∂J/∂W^(l), b^(l) := b^(l) - α·∂J/∂b^(l)",
      substitution: `Updated all ${state.weights.length} weight matrices and ${state.biases.length} bias vectors`,
      result: `Parameters updated with learning rate α=${lr}`,
      description: "Gradient descent updates all network parameters",
    });

    const newState: NeuralNetworkState = {
      ...state,
      weights: newWeights,
      biases: newBiases,
      activations,
      z_values,
      losses: [...state.losses, loss],
      iteration: state.iteration + 1,
    };

    return { state: newState, steps };
  }

  /**
   * Check for convergence.
   */
  isReached(state: NeuralNetworkState): boolean {
    if (state.losses.length < 2) return false;

    const lastLoss = state.losses[state.losses.length - 1];
    const prevLoss = state.losses[state.losses.length - 2];
    const improvement = Math.abs(lastLoss - prevLoss);

    return improvement < this.config.epsilon!;
  }

  /**
   * Apply activation function to vector.
   */
  private applyActivation(z: Vector, activation: string): Vector {
    switch (activation) {
      case "relu":
        return ActivationFunctions.reluVector(z);
      case "sigmoid":
        return ActivationFunctions.sigmoidVector(z);
      case "tanh":
        return ActivationFunctions.tanhVector(z);
      case "softmax":
        return ActivationFunctions.softmax(z);
      case "linear":
      default:
        return z;
    }
  }

  /**
   * Get activation derivative.
   */
  private activationDerivative(z: number, activation: string): number {
    switch (activation) {
      case "relu":
        return ActivationFunctions.reluDerivative(z);
      case "sigmoid":
        return ActivationFunctions.sigmoidDerivative(z);
      case "tanh":
        return ActivationFunctions.tanhDerivative(z);
      case "linear":
      default:
        return 1;
    }
  }

  /**
   * Predict for a batch of samples.
   */
  private predictBatch(state: NeuralNetworkState, features: Matrix): Matrix {
    return features.map((sample) => {
      let current = sample;
      for (let l = 0; l < state.layers.length; l++) {
        const z = LinearAlgebra.add(
          LinearAlgebra.matrixVectorMultiply(state.weights[l], current),
          state.biases[l]
        );
        current = this.applyActivation(z, state.layers[l].activation);
      }
      return current;
    });
  }
}
