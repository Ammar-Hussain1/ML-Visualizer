import { Vector } from "../types";

/**
 * Utility functions for decision trees
 */

export const TreeUtils = {
  /**
   * Compute entropy (measure of disorder)
   * H = -Σ p_i * log2(p_i)
   */
  entropy(labels: Vector): number {
    if (labels.length === 0) return 0;

    const counts: Record<number, number> = {};
    for (const label of labels) {
      counts[label] = (counts[label] || 0) + 1;
    }

    let entropy = 0;
    for (const count of Object.values(counts)) {
      const p = count / labels.length;
      if (p > 0) {
        entropy -= p * Math.log2(p);
      }
    }

    return entropy;
  },

  /**
   * Gini impurity: 1 - Σ p_i^2
   */
  gini(labels: Vector): number {
    if (labels.length === 0) return 0;

    const counts: Record<number, number> = {};
    for (const label of labels) {
      counts[label] = (counts[label] || 0) + 1;
    }

    let gini = 1;
    for (const count of Object.values(counts)) {
      const p = count / labels.length;
      gini -= p * p;
    }

    return gini;
  },

  /**
   * Information gain from a split
   * IG = entropy(parent) - weighted_avg_entropy(children)
   */
  informationGain(
    parentLabels: Vector,
    leftLabels: Vector,
    rightLabels: Vector
  ): number {
    const parentEntropy = this.entropy(parentLabels);
    const leftWeight = leftLabels.length / parentLabels.length;
    const rightWeight = rightLabels.length / parentLabels.length;

    const childrenEntropy =
      leftWeight * this.entropy(leftLabels) +
      rightWeight * this.entropy(rightLabels);

    return parentEntropy - childrenEntropy;
  },

  /**
   * Find best split for a feature
   * Returns: { threshold, infoGain, leftIndices, rightIndices }
   */
  findBestSplit(
    featureValues: Vector,
    labels: Vector
  ): {
    threshold: number;
    gain: number;
    leftIdx: number[];
    rightIdx: number[];
  } | null {
    const n = featureValues.length;
    if (n < 2) return null;

    // Sort indices by feature values
    const indices = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => featureValues[a] - featureValues[b]
    );

    let bestGain = -Infinity;
    let bestThreshold = -Infinity;
    let bestLeft: number[] = [];
    let bestRight: number[] = [];

    // Try split points between each pair of unique values
    const sortedValues = indices.map((i) => featureValues[i]);
    const uniqueValues = [...new Set(sortedValues)];

    for (let i = 0; i < uniqueValues.length - 1; i++) {
      const threshold =
        (uniqueValues[i] + uniqueValues[i + 1]) / 2;

      const leftIdx = indices.filter((i) => featureValues[i] <= threshold);
      const rightIdx = indices.filter((i) => featureValues[i] > threshold);

      if (leftIdx.length === 0 || rightIdx.length === 0) continue;

      const leftLabels = leftIdx.map((i) => labels[i]);
      const rightLabels = rightIdx.map((i) => labels[i]);

      const gain = this.informationGain(labels, leftLabels, rightLabels);

      if (gain > bestGain) {
        bestGain = gain;
        bestThreshold = threshold;
        bestLeft = leftIdx;
        bestRight = rightIdx;
      }
    }

    if (bestGain === -Infinity) return null;

    return {
      threshold: bestThreshold,
      gain: bestGain,
      leftIdx: bestLeft,
      rightIdx: bestRight,
    };
  },

  /**
   * Get majority class
   */
  majorityClass(labels: Vector): number {
    const counts: Record<number, number> = {};
    for (const label of labels) {
      counts[label] = (counts[label] || 0) + 1;
    }

    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0] as any;
  },
};
