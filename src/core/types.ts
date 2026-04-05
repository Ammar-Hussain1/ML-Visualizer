/**
 * Core type definitions for the ML Engine.
 * Following the 'No Black Box' philosophy, every computation 
 * uses these primitive types for full transparency.
 */

export type Vector = number[];
export type Matrix = number[][];

export interface Point {
  x: number;
  y: number;
  label?: string | number;
}

export interface TrainingData {
  features: Matrix;
  targets: Vector;
}

/**
 * Represents a single atomic step in an algorithm's execution.
 * Critical for the 'Explainable Math' engine.
 */
export interface Step {
  id: string;
  label: string;
  formula: string;       // LaTeX or symbolic representation
  substitution: string;  // Formula with actual numbers inserted
  result: string;        // The computed value
  description?: string;  // Human-readable context
}
