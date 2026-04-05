import { create } from "zustand";
import { Step } from "./types";

/**
 * Global State Management for the ML Visualizer.
 * Uses Zustand for simple, reactive state management.
 * 
 * Responsibilities:
 * - Current algorithm state
 * - List of executed steps
 * - Current step index for visualization
 * - Configuration and hyperparameters
 */

interface VisualizerState {
  // Algorithm execution
  currentAlgorithm: string | null;
  algorithmState: Record<string, unknown>;
  steps: Step[];
  currentStepIndex: number;
  
  // Configuration
  config: Record<string, unknown>;
  
  // UI state
  isTraining: boolean;
  isPaused: boolean;
  error: string | null;

  // Actions
  setCurrentAlgorithm: (algorithm: string) => void;
  setAlgorithmState: (state: Record<string, unknown>) => void;
  setSteps: (steps: Step[]) => void;
  addStep: (step: Step) => void;
  setCurrentStepIndex: (index: number) => void;
  setConfig: (config: Record<string, unknown>) => void;
  setIsTraining: (isTraining: boolean) => void;
  setIsPaused: (isPaused: boolean) => void;
  setError: (error: string | null) => void;
  
  // Utility actions
  nextStep: () => void;
  previousStep: () => void;
  resetVisualization: () => void;
}

export const useVisualizerStore = create<VisualizerState>((set) => ({
  currentAlgorithm: null,
  algorithmState: {},
  steps: [],
  currentStepIndex: -1,
  config: {},
  isTraining: false,
  isPaused: false,
  error: null,

  setCurrentAlgorithm: (algorithm: string) =>
    set({ currentAlgorithm: algorithm }),

  setAlgorithmState: (state: Record<string, unknown>) =>
    set({ algorithmState: state }),

  setSteps: (steps: Step[]) => set({ steps }),

  addStep: (step: Step) =>
    set((state) => ({
      steps: [...state.steps, step],
      currentStepIndex: state.steps.length,
    })),

  setCurrentStepIndex: (index: number) =>
    set({ currentStepIndex: index }),

  setConfig: (config: Record<string, unknown>) =>
    set({ config }),

  setIsTraining: (isTraining: boolean) =>
    set({ isTraining }),

  setIsPaused: (isPaused: boolean) =>
    set({ isPaused }),

  setError: (error: string | null) =>
    set({ error }),

  nextStep: () =>
    set((state) => ({
      currentStepIndex: Math.min(state.currentStepIndex + 1, state.steps.length - 1),
    })),

  previousStep: () =>
    set((state) => ({
      currentStepIndex: Math.max(state.currentStepIndex - 1, -1),
    })),

  resetVisualization: () =>
    set({
      currentAlgorithm: null,
      algorithmState: {},
      steps: [],
      currentStepIndex: -1,
      isTraining: false,
      isPaused: false,
      error: null,
    }),
}));
