import { Vector } from "../types";

/**
 * Distance Metrics for KNN and clustering
 */

export const DistanceMetrics = {
  /**
   * Euclidean distance: sqrt(sum((xi - yi)^2))
   */
  euclidean(p1: Vector, p2: Vector): number {
    if (p1.length !== p2.length) {
      throw new Error("Vector dimension mismatch");
    }
    let sum = 0;
    for (let i = 0; i < p1.length; i++) {
      const diff = p1[i] - p2[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  },

  /**
   * Manhattan distance (L1): sum(|xi - yi|)
   */
  manhattan(p1: Vector, p2: Vector): number {
    if (p1.length !== p2.length) {
      throw new Error("Vector dimension mismatch");
    }
    let sum = 0;
    for (let i = 0; i < p1.length; i++) {
      sum += Math.abs(p1[i] - p2[i]);
    }
    return sum;
  },

  /**
   * Minkowski distance: (sum(|xi - yi|^p))^(1/p)
   */
  minkowski(p1: Vector, p2: Vector, p: number = 2): number {
    if (p1.length !== p2.length) {
      throw new Error("Vector dimension mismatch");
    }
    if (p === Infinity) {
      return Math.max(...p1.map((x, i) => Math.abs(x - p2[i])));
    }
    let sum = 0;
    for (let i = 0; i < p1.length; i++) {
      sum += Math.pow(Math.abs(p1[i] - p2[i]), p);
    }
    return Math.pow(sum, 1 / p);
  },

  /**
   * Cosine distance: 1 - (u·v / (||u|| * ||v||))
   */
  cosine(p1: Vector, p2: Vector): number {
    if (p1.length !== p2.length) {
      throw new Error("Vector dimension mismatch");
    }
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < p1.length; i++) {
      dotProduct += p1[i] * p2[i];
      norm1 += p1[i] * p1[i];
      norm2 += p2[i] * p2[i];
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    if (norm1 === 0 || norm2 === 0) return 1;

    return 1 - dotProduct / (norm1 * norm2);
  },

  /**
   * Hamming distance: number of positions where values differ (binary)
   */
  hamming(p1: Vector, p2: Vector): number {
    if (p1.length !== p2.length) {
      throw new Error("Vector dimension mismatch");
    }
    let count = 0;
    for (let i = 0; i < p1.length; i++) {
      if (p1[i] !== p2[i]) count++;
    }
    return count;
  },
};
