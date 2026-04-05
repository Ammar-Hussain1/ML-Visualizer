import { Step } from "./types";

/**
 * Step Engine: Core deterministic execution system.
 * 
 * The Step Engine is responsible for:
 * - Executing steps sequentially
 * - Maintaining execution history
 * - Providing undo/redo capabilities
 * - Tracking state changes
 * 
 * Every algorithm must emit steps through this engine to ensure
 * full traceability and the ability to visualize every computation.
 */

export interface ExecutionContext {
  stepIndex: number;
  history: Step[];
  future: Step[]; // For redo functionality
  currentState: Record<string, unknown>;
}

export class StepEngine {
  private context: ExecutionContext;

  constructor(initialState: Record<string, unknown> = {}) {
    this.context = {
      stepIndex: -1,
      history: [],
      future: [],
      currentState: initialState,
    };
  }

  /**
   * Execute a single step forward.
   */
  executeStep(step: Step, stateUpdate?: Record<string, unknown>): void {
    // Clear future on new execution
    if (this.context.future.length > 0) {
      this.context.future = [];
    }

    this.context.history.push(step);
    this.context.stepIndex++;

    if (stateUpdate) {
      this.context.currentState = {
        ...this.context.currentState,
        ...stateUpdate,
      };
    }
  }

  /**
   * Execute multiple steps at once.
   */
  executeSteps(
    steps: Step[],
    stateUpdates?: Record<string, unknown>[]
  ): void {
    steps.forEach((step, index) => {
      const update = stateUpdates?.[index];
      this.executeStep(step, update);
    });
  }

  /**
   * Undo the last step.
   */
  undo(): Step | null {
    if (this.context.stepIndex < 0) return null;

    const step = this.context.history[this.context.stepIndex];
    this.context.future.unshift(step);
    this.context.stepIndex--;
    
    return step;
  }

  /**
   * Redo the last undone step.
   */
  redo(): Step | null {
    if (this.context.future.length === 0) return null;

    const step = this.context.future.shift()!;
    this.context.history.push(step);
    this.context.stepIndex++;

    return step;
  }

  /**
   * Reset to initial state.
   */
  reset(initialState: Record<string, unknown> = {}): void {
    this.context = {
      stepIndex: -1,
      history: [],
      future: [],
      currentState: initialState,
    };
  }

  /**
   * Get the current execution context.
   */
  getContext(): ExecutionContext {
    return { ...this.context };
  }

  /**
   * Get all executed steps up to current position.
   */
  getExecutedSteps(): Step[] {
    return this.context.history.slice(0, this.context.stepIndex + 1);
  }

  /**
   * Get the current step (if any).
   */
  getCurrentStep(): Step | null {
    if (this.context.stepIndex < 0) return null;
    return this.context.history[this.context.stepIndex] || null;
  }

  /**
   * Get current state.
   */
  getState(): Record<string, unknown> {
    return { ...this.context.currentState };
  }

  /**
   * Check if we can undo.
   */
  canUndo(): boolean {
    return this.context.stepIndex >= 0;
  }

  /**
   * Check if we can redo.
   */
  canRedo(): boolean {
    return this.context.future.length > 0;
  }

  /**
   * Get total steps executed.
   */
  getTotalSteps(): number {
    return this.context.stepIndex + 1;
  }
}
