import { Algorithm, AlgorithmConfig } from "../algorithm";
import { Step, Vector, Matrix } from "../types";
import { TreeUtils } from "../math/tree-utils";

/**
 * Decision Tree Node
 */
export interface TreeNode {
  isLeaf: boolean;
  threshold?: number; // For split decisions
  featureIdx?: number; // Which feature to split on
  prediction?: number; // For leaf nodes
  left?: TreeNode;
  right?: TreeNode;
  samples: number; // Number of samples at this node
  gini: number; // Impurity at this node
}

/**
 * Decision Tree State
 */
export interface DecisionTreeState {
  features: Matrix;
  targets: Vector;
  tree?: TreeNode;
  maxDepth: number;
  minSamplesSplit: number;
  predictions?: Vector;
  accuracies?: number[];
  iteration: number;
  losses: number[];
}

/**
 * Decision Tree Classifier
 * 
 * Algorithm:
 * 1. Recursively find best feature and threshold to split on (maximize information gain)
 * 2. Split dataset and repeat for each child
 * 3. Stop when: max depth reached, min samples, or all samples same class
 * 4. Leaf nodes use majority class for prediction
 */

export class DecisionTree implements Algorithm<DecisionTreeState> {
  name = "Decision Tree";
  config: AlgorithmConfig;
  private maxDepth: number;
  private minSamplesSplit: number;

  constructor(
    maxDepth: number = 5,
    minSamplesSplit: number = 2,
    config: AlgorithmConfig = {}
  ) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.config = {
      learningRate: 0,
      maxIterations: 1,
      epsilon: 0.01,
      ...config,
    };
  }

  /**
   * Build the decision tree during initialization
   */
  initialize(features: Matrix, targets: Vector): DecisionTreeState {
    const state: DecisionTreeState = {
      features,
      targets,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      iteration: 0,
      losses: [],
    };

    // Build tree
    const indices = Array.from({ length: features.length }, (_, i) => i);
    state.tree = this.buildTree(
      features,
      targets,
      indices,
      0,
      []
    );

    return state;
  }

  /**
   * Forward pass: predict on a single sample
   */
  forward(state: DecisionTreeState, input: Vector): { result: number; steps: Step[] } {
    const steps: Step[] = [];

    if (!state.tree) {
      return { result: 0, steps };
    }

    let node = state.tree;
    let depth = 0;

    while (!node.isLeaf) {
      depth++;
      const value = input[node.featureIdx!];
      const threshold = node.threshold!;

      steps.push({
        id: `traverse-${depth}`,
        label: `Decision Node ${depth}`,
        formula: `if x_${node.featureIdx} <= ${threshold.toFixed(3)} then go left else go right`,
        substitution: `x_${node.featureIdx} = ${value.toFixed(3)} ${value <= threshold ? "<=" : ">"} ${threshold.toFixed(3)}`,
        result: `Going ${value <= threshold ? "LEFT" : "RIGHT"}`,
        description: `Node checks feature ${node.featureIdx} against threshold`,
      });

      node = value <= threshold ? node.left! : node.right!;
    }

    steps.push({
      id: "predict",
      label: "Leaf Node Prediction",
      formula: `prediction = majority_class(leaf_samples)`,
      substitution: `Reached leaf node with ${node.samples} samples`,
      result: `Predicted Class: ${node.prediction}`,
      description: "Return majority class of leaf node",
    });

    return { result: node.prediction!, steps };
  }

  /**
   * Training: evaluate on test set
   */
  update(state: DecisionTreeState): { state: DecisionTreeState; steps: Step[] } {
    const steps: Step[] = [];

    if (!state.tree) {
      return { state, steps };
    }

    // Evaluate accuracy on training set
    let correct = 0;
    for (let i = 0; i < state.features.length; i++) {
      const { result: prediction } = this.forward(state, state.features[i]);
      if (prediction === state.targets[i]) correct++;
    }

    const accuracy = (correct / state.features.length) * 100;
    const loss = 1 - correct / state.features.length;

    steps.push({
      id: `iter-${state.iteration}-1`,
      label: "Evaluate on Training Set",
      formula: `accuracy = (# correct / # total) × 100%`,
      substitution: `Tested ${state.features.length} samples`,
      result: `Accuracy: ${accuracy.toFixed(2)}%, Correct: ${correct}/${state.features.length}`,
      description: "Decision trees perfectly fit training data by construction",
    });

    steps.push({
      id: `iter-${state.iteration}-2`,
      label: "Tree Statistics",
      formula: `tree_depth ≤ max_depth, samples ≥ min_samples_split`,
      substitution: `Max depth: ${this.maxDepth}, Min samples: ${this.minSamplesSplit}`,
      result: `Tree built with optimal splits`,
      description: "Decision tree training complete",
    });

    const newState: DecisionTreeState = {
      ...state,
      losses: [...state.losses, loss],
      accuracies: [...(state.accuracies ?? []), accuracy],
      iteration: state.iteration + 1,
    };

    return { state: newState, steps };
  }

  /**
   * Check convergence - trees converge after first iteration
   */
  isReached(state: DecisionTreeState): boolean {
    return state.iteration > 0;
  }

  /**
   * Recursively build the decision tree
   */
  private buildTree(
    features: Matrix,
    targets: Vector,
    indices: number[],
    depth: number,
    steps: any[]
  ): TreeNode {
    const labels = indices.map((i) => targets[i]);
    const giniImpurity = TreeUtils.gini(labels);

    // Check stopping criteria
    if (
      depth >= this.maxDepth ||
      indices.length < this.minSamplesSplit ||
      giniImpurity === 0
    ) {
      // Leaf node
      return {
        isLeaf: true,
        prediction: TreeUtils.majorityClass(labels),
        samples: indices.length,
        gini: giniImpurity,
      };
    }

    // Find best split
    let bestGain = -Infinity;
    let bestFeature = 0;
    let bestSplit: {
      threshold: number;
      gain: number;
      leftIdx: number[];
      rightIdx: number[];
    } | null = null;

    // Try each feature
    for (let f = 0; f < features[0].length; f++) {
      const featureValues = indices.map((i) => features[i][f]);
      const split = TreeUtils.findBestSplit(featureValues, labels);

      if (split && split.gain > bestGain) {
        bestGain = split.gain;
        bestFeature = f;
        bestSplit = split;
      }
    }

    // If no good split found, return leaf
    if (!bestSplit || bestGain < 0.001) {
      return {
        isLeaf: true,
        prediction: TreeUtils.majorityClass(labels),
        samples: indices.length,
        gini: giniImpurity,
      };
    }

    // Recursive split
    const leftIndices = bestSplit.leftIdx.map((i) => indices[i]);
    const rightIndices = bestSplit.rightIdx.map((i) => indices[i]);

    return {
      isLeaf: false,
      threshold: bestSplit.threshold,
      featureIdx: bestFeature,
      samples: indices.length,
      gini: giniImpurity,
      left: this.buildTree(
        features,
        targets,
        leftIndices,
        depth + 1,
        steps
      ),
      right: this.buildTree(
        features,
        targets,
        rightIndices,
        depth + 1,
        steps
      ),
    };
  }
}
