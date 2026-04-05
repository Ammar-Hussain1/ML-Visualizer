import { Algorithm, AlgorithmConfig } from "./algorithm";
import { Step, TrainingData, Vector, Matrix } from "./types";
import { StepEngine } from "./step-engine";

/**
 * Training Engine: Orchestrates algorithm execution.
 * 
 * Responsibilities:
 * - Manages training iterations
 * - Collects steps from algorithms
 * - Tracks convergence
 * - Provides callbacks for visualization
 */

export interface TrainingProgress {
  iteration: number;
  totalIterations: number;
  steps: Step[];
  isConverged: boolean;
  currentState: Record<string, unknown>;
}

export interface TrainingCallbacks {
  onStepComplete?: (step: Step, progress: TrainingProgress) => void;
  onIterationComplete?: (progress: TrainingProgress) => void;
  onTrainingComplete?: (progress: TrainingProgress) => void;
  onError?: (error: Error) => void;
}

export class TrainingEngine {
  private stepEngine: StepEngine;
  private algorithm: Algorithm;
  private callbacks: TrainingCallbacks;
  private isRunning = false;

  constructor(
    algorithm: Algorithm,
    callbacks: TrainingCallbacks = {}
  ) {
    this.algorithm = algorithm;
    this.callbacks = callbacks;
    this.stepEngine = new StepEngine();
  }

  /**
   * Initialize the algorithm with training data.
   */
  initialize(trainingData: Matrix, targets: Vector): Record<string, unknown> {
    const state = this.algorithm.initialize(trainingData, targets);
    this.stepEngine.reset({ algorithmState: state });
    return state as Record<string, unknown>;
  }

  /**
   * Run a single training iteration.
   */
  trainIteration(
    state: unknown,
    iterationNum: number,
    totalIterations: number
  ): { state: unknown; steps: Step[] } {
    const result = this.algorithm.update(state);

    // Execute steps through the step engine
    result.steps.forEach((step) => {
      this.stepEngine.executeStep(step);
      this.callbacks.onStepComplete?.(step, this.getProgress(iterationNum, totalIterations));
    });

    this.callbacks.onIterationComplete?.(
      this.getProgress(iterationNum, totalIterations)
    );

    return result;
  }

  /**
   * Run full training loop until convergence or max iterations.
   */
  async train(
    maxIterations: number = 100,
    verbose = false
  ): Promise<Record<string, unknown>> {
    if (this.isRunning) {
      throw new Error("Training is already in progress");
    }

    this.isRunning = true;
    let state = this.stepEngine.getState().algorithmState as unknown;

    try {
      for (let i = 0; i < maxIterations; i++) {
        const result = this.trainIteration(state, i + 1, maxIterations);
        state = result.state;

        if (verbose) {
          console.log(`Iteration ${i + 1}/${maxIterations}`);
        }

        // Check for convergence
        if (this.algorithm.isReached(state)) {
          if (verbose) console.log("Converged!");
          break;
        }

        // Allow async operations (visualization updates)
        await new Promise((resolve) => setTimeout(resolve, 0));
      }

      this.callbacks.onTrainingComplete?.(this.getProgress(maxIterations, maxIterations));
      return state as Record<string, unknown>;
    } catch (error) {
      this.callbacks.onError?.(error as Error);
      throw error;
    } finally {
      this.isRunning = false;
    }
  }

  /**
   * Get current training progress.
   */
  private getProgress(iteration: number, totalIterations: number): TrainingProgress {
    return {
      iteration,
      totalIterations,
      steps: this.stepEngine.getExecutedSteps(),
      isConverged: this.algorithm.isReached(
        this.stepEngine.getState().algorithmState as unknown
      ),
      currentState: this.stepEngine.getState(),
    };
  }

  /**
   * Get all executed steps.
   */
  getSteps(): Step[] {
    return this.stepEngine.getExecutedSteps();
  }

  /**
   * Get the step engine for manual control.
   */
  getStepEngine(): StepEngine {
    return this.stepEngine;
  }

  /**
   * Check if training is currently running.
   */
  isTraining(): boolean {
    return this.isRunning;
  }

  /**
   * Reset training state.
   */
  reset(): void {
    this.stepEngine.reset();
  }
}
