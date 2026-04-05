import { Matrix, Vector, Step } from "./types";

/**
 * Common configuration for all ML algorithms.
 */
export interface AlgorithmConfig {
  learningRate?: number;
  maxIterations?: number;
  epsilon?: number;
}

/**
 * Universal interface for all Machine Learning algorithms.
 * Guarantees each module is plug-and-play, deterministic, and traceable.
 */
export interface Algorithm<TState = unknown, TResult = unknown> {
  name: string;
  config: AlgorithmConfig;

  /**
   * Initializes the internal state with configuration.
   */
  initialize(trainingData: Matrix, targets: Vector): TState;

  /**
   * Executes a single forward movement (e.g., a prediction).
   */
  forward(state: TState, input: Vector): { result: TResult; steps: Step[] };

  /**
   * Performs an update/training step.
   * Returns the new state and the computation steps taken.
   */
  update(state: TState): { state: TState; steps: Step[] };

  /**
   * Checks for convergence.
   */
  isReached(state: TState): boolean;
}
