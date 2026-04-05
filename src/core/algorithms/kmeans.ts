import { Algorithm, AlgorithmConfig } from "../algorithm";
import { Step, Matrix, Vector } from "../types";
import { DistanceMetrics } from "../math/distance-metrics";

/**
 * Compute mean (centroid) of a set of points
 * Returns the mean along each dimension
 */
function computeCentroid(points: Matrix): Vector {
  if (points.length === 0) return [];
  const dims = points[0].length;
  const centroid: Vector = [];

  for (let d = 0; d < dims; d++) {
    let sum = 0;
    for (let i = 0; i < points.length; i++) {
      sum += points[i][d];
    }
    centroid.push(sum / points.length);
  }

  return centroid;
}

/**
 * K-Means Clustering State
 * 
 * Tracks:
 * - Current cluster centroids
 * - Data point assignments
 * - Inertia (sum of squared distances to nearest centroid)
 * - Iteration history
 */
export interface KMeansState {
  features: Matrix;
  k: number;
  centroids: Matrix;
  assignments: Vector;
  inertia: number;
  iteration: number;
  inertias: number[];
  centroidHistory: Matrix[];
}

/**
 * K-Means Clustering Algorithm
 * 
 * Algorithm:
 * 1. Initialize k random centroids
 * 2. Assign each point to nearest centroid
 * 3. Recompute centroids as mean of assigned points
 * 4. Repeat until convergence
 * 
 * Time Complexity: O(n*k*d*i) where n=samples, k=clusters, d=dimensions, i=iterations
 * Space Complexity: O(n*k*d)
 */
export class KMeans implements Algorithm<KMeansState, number> {
  name = "K-Means Clustering";
  config: AlgorithmConfig;
  private k: number;
  private maxIterations: number;
  private tolerance: number;

  constructor(
    k: number = 3,
    config: AlgorithmConfig = {}
  ) {
    this.k = k;
    this.config = config;
    this.maxIterations = config.maxIterations || 100;
    this.tolerance = 1e-4; // Convergence threshold for inertia change
  }

  /**
   * Initialize K-means state with random centroids
   * 
   * Formula:
   * centroids = random k points from feature space
   */
  initialize(features: Matrix, targets?: Vector): KMeansState {
    const n = features.length;
    const d = features[0].length;

    // Randomly select k points as initial centroids
    const centroids: Matrix = [];
    const selectedIndices = new Set<number>();

    while (centroids.length < this.k) {
      const idx = Math.floor(Math.random() * n);
      if (!selectedIndices.has(idx)) {
        centroids.push([...features[idx]]);
        selectedIndices.add(idx);
      }
    }

    // Initial assignments (all to first centroid for now)
    const assignments = Array(n).fill(0);

    // Compute initial inertia
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      const dist = DistanceMetrics.euclidean(features[i], centroids[0]);
      inertia += dist * dist;
    }

    return {
      features,
      k: this.k,
      centroids,
      assignments,
      inertia,
      iteration: 0,
      inertias: [inertia],
      centroidHistory: [JSON.parse(JSON.stringify(centroids))],
    };
  }

  /**
   * Forward pass: Assign a single point to nearest centroid
   * 
   * For a given point x_i:
   * c_i = argmin_j ||x_i - μ_j||²
   */
  forward(state: KMeansState, input: Vector): { result: number; steps: Step[] } {
    const steps: Step[] = [];
    const { centroids, k } = state;
    
    let minDist = Infinity;
    let closestCentroid = 0;

    for (let j = 0; j < k; j++) {
      const dist = DistanceMetrics.euclidean(input, centroids[j]);
      if (dist < minDist) {
        minDist = dist;
        closestCentroid = j;
      }
    }

    steps.push({
      id: `predict-cluster`,
      label: `Assign point to nearest centroid`,
      formula: `c = argmin_j ||x - μ_j||²`,
      substitution: `Min distance: ${minDist.toFixed(3)}`,
      result: `Cluster ${closestCentroid}`,
    });

    return { result: closestCentroid, steps };
  }

  /**
   * Update pass: Recompute centroids and check convergence
   * 
   * For each cluster j:
   * μ_j = (1/n_j) * Σ(x_i : c_i = j)
   * 
   * where n_j is the number of points in cluster j
   */
  update(state: KMeansState): { state: KMeansState; steps: Step[] } {
    const steps: Step[] = [];
    const { features, centroids, assignments, k, iteration } = state;

    // Assign points to nearest centroids
    const newAssignments: Vector = [];
    for (let i = 0; i < features.length; i++) {
      let minDist = Infinity;
      let closestCentroid = 0;

      for (let j = 0; j < k; j++) {
        const dist = DistanceMetrics.euclidean(features[i], centroids[j]);
        if (dist < minDist) {
          minDist = dist;
          closestCentroid = j;
        }
      }

      newAssignments.push(closestCentroid);
    }

    // Recompute centroids
    const newCentroids: Matrix = [];
    for (let j = 0; j < k; j++) {
      const clusterPoints: Matrix = [];
      for (let i = 0; i < features.length; i++) {
        if (newAssignments[i] === j) {
          clusterPoints.push(features[i]);
        }
      }

      if (clusterPoints.length > 0) {
        newCentroids.push(computeCentroid(clusterPoints));
      } else {
        // Keep old centroid if cluster is empty
        newCentroids.push([...centroids[j]]);
      }

      steps.push({
        id: `update-centroid-${j}`,
        label: `Update centroid ${j + 1}`,
        formula: `μ_${j} = (1/n_${j}) * Σ x_i`,
        substitution: `n_${j} = ${clusterPoints.length} points`,
        result: `μ_${j} = [${newCentroids[j]
          .map((v) => v.toFixed(3))
          .join(", ")}]`,
      });
    }

    // Compute inertia (within-cluster sum of squares)
    let newInertia = 0;
    for (let i = 0; i < features.length; i++) {
      const clusterIdx = newAssignments[i];
      const dist = DistanceMetrics.euclidean(features[i], newCentroids[clusterIdx]);
      newInertia += dist * dist;
    }

    steps.push({
      id: `compute-inertia`,
      label: `Compute within-cluster inertia`,
      formula: `J = Σ_i ||x_i - μ_{c_i}||²`,
      substitution: `Sum of squared distances`,
      result: `J = ${newInertia.toFixed(4)}`,
    });

    const centroidHistory = [...state.centroidHistory];
    centroidHistory.push(JSON.parse(JSON.stringify(newCentroids)));

    const newState: KMeansState = {
      ...state,
      centroids: newCentroids,
      assignments: newAssignments,
      inertia: newInertia,
      iteration: iteration + 1,
      inertias: [...state.inertias, newInertia],
      centroidHistory,
    };

    return { state: newState, steps };
  }

  /**
   * Check convergence: Stop if inertia change < tolerance or max iterations reached
   */
  isReached(state: KMeansState): boolean {
    const { iteration, inertias } = state;

    if (iteration >= this.maxIterations) {
      return true;
    }

    if (inertias.length > 1) {
      const inertiaChange = Math.abs(
        inertias[inertias.length - 1] - inertias[inertias.length - 2]
      );
      if (inertiaChange < this.tolerance) {
        return true;
      }
    }

    return false;
  }
}
