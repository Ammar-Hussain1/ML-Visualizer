import { Algorithm, AlgorithmConfig } from "../algorithm";
import { Step, Matrix, Vector } from "../types";
import { LinearAlgebra } from "../math/linear-algebra";

/**
 * PCA State Interface
 * Tracks principal components, explained variance, and projection history
 */
export interface PCAState {
  features: Matrix;
  numComponents: number;
  mean: Vector;
  components: Matrix; // Principal components (eigenvectors)
  variance: Vector; // Variance explained by each component
  cumulativeVariance: number[];
  projections?: Matrix; // Projected features onto principal components
  iteration: number;
  losses: number[]; // Reconstruction error
}

/**
 * Principal Component Analysis (PCA)
 * 
 * Algorithm:
 * 1. Center the data (subtract mean)
 * 2. Compute covariance matrix
 * 3. Find eigenvalues and eigenvectors of covariance matrix
 * 4. Sort by eigenvalues (variance explained)
 * 5. Select top k eigenvectors as principal components
 * 6. Project data onto principal components
 * 
 * Time Complexity: O(d³) for eigendecomposition where d = number of features
 * Space Complexity: O(d²) for covariance matrix
 */
export class PCA implements Algorithm<PCAState, Vector> {
  name = "Principal Component Analysis";
  config: AlgorithmConfig;
  private numComponents: number;

  constructor(numComponents: number = 2, config: AlgorithmConfig = {}) {
    this.numComponents = numComponents;
    this.config = {
      learningRate: 0, // PCA doesn't use learning rate
      maxIterations: 1, // PCA only needs 1 iteration
      ...config,
    };
  }

  /**
   * Initialize PCA by computing principal components
   * 
   * Steps:
   * 1. Compute mean of features
   * 2. Center data
   * 3. Compute covariance matrix
   * 4. Perform eigendecomposition
   * 5. Select top components
   */
  initialize(features: Matrix, targets?: Vector): PCAState {
    const n = features.length;
    const d = features[0].length;

    // Step 1: Compute mean
    const mean: Vector = [];
    for (let j = 0; j < d; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += features[i][j];
      }
      mean.push(sum / n);
    }

    // Step 2: Center data
    const centered: Matrix = [];
    for (let i = 0; i < n; i++) {
      const row: Vector = [];
      for (let j = 0; j < d; j++) {
        row.push(features[i][j] - mean[j]);
      }
      centered.push(row);
    }

    // Step 3: Compute covariance matrix
    // Cov = (1/n) * X^T * X (for centered X)
    const cov: Matrix = Array(d)
      .fill(0)
      .map(() => Array(d).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        for (let k = 0; k < d; k++) {
          cov[j][k] += (centered[i][j] * centered[i][k]) / n;
        }
      }
    }

    // Step 4: Compute eigendecomposition using Power Iteration
    // For simplicity and determinism, we'll use a simplified approach
    const eigenPairs = this.powerIteration(cov, Math.min(this.numComponents, d));

    // Step 5: Sort by variance (eigenvalues) in descending order
    eigenPairs.sort((a, b) => b.eigenvalue - a.eigenvalue);

    // Step 6: Extract components and variance
    const components: Matrix = eigenPairs.map((pair) => pair.eigenvector);
    const variance: Vector = eigenPairs.map((pair) => pair.eigenvalue);

    // Compute cumulative variance
    const totalVariance = variance.reduce((a, b) => a + b, 0);
    const cumulativeVariance: number[] = [];
    let cumSum = 0;
    for (const v of variance) {
      cumSum += v / totalVariance;
      cumulativeVariance.push(cumSum);
    }

    // Project data onto principal components
    const projections: Matrix = [];
    for (let i = 0; i < n; i++) {
      const proj: Vector = [];
      for (let j = 0; j < this.numComponents; j++) {
        let sum = 0;
        for (let k = 0; k < d; k++) {
          sum += centered[i][k] * components[j][k];
        }
        proj.push(sum);
      }
      projections.push(proj);
    }

    // Compute reconstruction error
    let reconstructionError = 0;
    for (let i = 0; i < n; i++) {
      // Project back to original space
      const reconstructed: Vector = Array(d).fill(0);
      for (let j = 0; j < this.numComponents; j++) {
        for (let k = 0; k < d; k++) {
          reconstructed[k] += projections[i][j] * components[j][k];
        }
      }

      // Compute error
      for (let k = 0; k < d; k++) {
        const error = centered[i][k] - reconstructed[k];
        reconstructionError += error * error;
      }
    }
    reconstructionError /= n;

    return {
      features,
      numComponents: this.numComponents,
      mean,
      components,
      variance,
      cumulativeVariance,
      projections,
      iteration: 0,
      losses: [reconstructionError],
    };
  }

  /**
   * Forward pass: Project a new point onto principal components
   */
  forward(state: PCAState, input: Vector): { result: Vector; steps: Step[] } {
    const steps: Step[] = [];
    const { mean, components, numComponents } = state;

    // Center the input
    const centered: Vector = input.map((x, i) => x - mean[i]);

    // Project onto principal components
    const projection: Vector = [];
    for (let j = 0; j < numComponents; j++) {
      let sum = 0;
      for (let k = 0; k < input.length; k++) {
        sum += centered[k] * components[j][k];
      }
      projection.push(sum);
    }

    steps.push({
      id: `center-input`,
      label: "Center input by subtracting mean",
      formula: `x_centered = x - μ`,
      substitution: `μ = [${mean.map((m) => m.toFixed(2)).join(", ")}]`,
      result: `x_centered = [${centered.map((c) => c.toFixed(2)).join(", ")}]`,
    });

    steps.push({
      id: `project`,
      label: `Project onto ${numComponents} principal components`,
      formula: `proj_j = Σ_k x_centered[k] * PC_j[k]`,
      substitution: `For each component j`,
      result: `projection = [${projection.map((p) => p.toFixed(2)).join(", ")}]`,
    });

    return { result: projection, steps };
  }

  /**
   * Update pass: Recompute statistics (not really used in PCA)
   * Since PCA is unsupervised, we just return the same state
   */
  update(state: PCAState): { state: PCAState; steps: Step[] } {
    const steps: Step[] = [
      {
        id: `pca-complete`,
        label: "PCA computation complete",
        formula: `Principal components already computed in initialization`,
        substitution: `Variance explained: ${(state.cumulativeVariance[state.numComponents - 1] * 100).toFixed(1)}%`,
        result: `PCA ready for projection`,
      },
    ];

    return { state, steps };
  }

  /**
   * Convergence check: PCA converges in 1 iteration
   */
  isReached(state: PCAState): boolean {
    return state.iteration >= 1;
  }

  /**
   * Power iteration method to find dominant eigenvector
   * Simplified for numerical stability
   */
  private powerIteration(
    matrix: Matrix,
    k: number
  ): Array<{ eigenvector: Vector; eigenvalue: number }> {
    const d = matrix.length;
    const eigenPairs: Array<{ eigenvector: Vector; eigenvalue: number }> = [];
    const M = JSON.parse(JSON.stringify(matrix)); // Deep copy

    for (let iter = 0; iter < k; iter++) {
      // Initialize random vector
      let v: Vector = Array(d)
        .fill(0)
        .map(() => Math.random());

      // Normalize
      let norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0));
      v = v.map((x) => x / norm);

      // Power iteration
      for (let i = 0; i < 100; i++) {
        // Mv
        let newV: Vector = Array(d).fill(0);
        for (let j = 0; j < d; j++) {
          for (let l = 0; l < d; l++) {
            newV[j] += M[j][l] * v[l];
          }
        }

        // Normalize
        norm = Math.sqrt(newV.reduce((a, b) => a + b * b, 0));
        if (norm > 1e-10) {
          newV = newV.map((x) => x / norm);
        }

        // Check convergence
        let diff = 0;
        for (let j = 0; j < d; j++) {
          diff += Math.abs(newV[j] - v[j]);
        }
        if (diff < 1e-6) break;

        v = newV;
      }

      // Compute eigenvalue: λ = v^T * A * v
      let eigenvalue = 0;
      let Av: Vector = Array(d).fill(0);
      for (let j = 0; j < d; j++) {
        for (let l = 0; l < d; l++) {
          Av[j] += M[j][l] * v[l];
        }
      }
      for (let j = 0; j < d; j++) {
        eigenvalue += v[j] * Av[j];
      }

      eigenPairs.push({ eigenvector: v, eigenvalue: Math.max(0, eigenvalue) });

      // Deflate matrix: M = M - λ * v * v^T
      for (let j = 0; j < d; j++) {
        for (let l = 0; l < d; l++) {
          M[j][l] -= eigenvalue * v[j] * v[l];
        }
      }
    }

    return eigenPairs;
  }
}
