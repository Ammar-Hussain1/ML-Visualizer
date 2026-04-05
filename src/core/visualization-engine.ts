import { Vector, Matrix, Point } from "./types";
import { LinearAlgebra } from "./math/linear-algebra";

/**
 * Visualization Engine: Transforms data for visual representation.
 * 
 * Responsibilities:
 * - Scale and normalize data for rendering
 * - Generate chart configurations
 * - Handle coordinate transformations
 * - Prepare data for different visualization types
 * 
 * Note: This engine produces data/config, not actual visuals.
 * Rendering is handled by React components.
 */

export interface ChartConfig {
  type: "scatter" | "line" | "matrix" | "histogram" | "heatmap" | "surface";
  title: string;
  data: unknown;
  axes?: {
    x: AxisConfig;
    y: AxisConfig;
  };
  theme?: {
    primaryColor: string;
    secondaryColor: string;
    accentColor: string;
  };
}

export interface AxisConfig {
  label: string;
  min: number;
  max: number;
  unit?: string;
  scale?: "linear" | "log";
}

export interface ScatterData {
  points: Point[];
  dimensions: number;
  clusters?: number[];
}

export interface LineData {
  x: Vector;
  y: Vector;
  label?: string;
  lines?: { x: Vector; y: Vector; label: string }[];
}

export interface MatrixData {
  matrix: Matrix;
  labels?: {
    rows: string[];
    cols: string[];
  };
}

export class VisualizationEngine {
  private defaultTheme = {
    primaryColor: "#3b82f6",
    secondaryColor: "#ef4444",
    accentColor: "#10b981",
  };

  /**
   * Generate scatter plot configuration from 2D data.
   */
  scatterPlot(
    data: Vector[],
    targets?: Vector,
    title = "Data Visualization"
  ): ChartConfig {
    if (data[0].length !== 2) {
      throw new Error("Scatter plot requires 2D data. Use dimensionality reduction for higher dimensions.");
    }

    const points: Point[] = data.map((point, i) => ({
      x: point[0],
      y: point[1],
      label: targets ? `Class ${targets[i]}` : undefined,
    }));

    const xValues = points.map((p) => p.x);
    const yValues = points.map((p) => p.y);

    return {
      type: "scatter",
      title,
      data: { points } as ScatterData,
      axes: {
        x: this.createAxis(xValues, "x₁", "Feature 1"),
        y: this.createAxis(yValues, "x₂", "Feature 2"),
      },
      theme: this.defaultTheme,
    };
  }

  /**
   * Generate line plot configuration for showing trends.
   */
  linePlot(
    x: Vector,
    y: Vector,
    title = "Loss Over Time",
    label = "Value"
  ): ChartConfig {
    return {
      type: "line",
      title,
      data: { x, y, label } as LineData,
      axes: {
        x: this.createAxis(x, "Iteration", "Training Iteration"),
        y: this.createAxis(y, "Loss", "Loss Value"),
      },
      theme: this.defaultTheme,
    };
  }

  /**
   * Generate configuration for showing decision boundary.
   */
  decisionBoundary(
    features: Matrix,
    predictions: Vector,
    targets?: Vector,
    title = "Decision Boundary"
  ): ChartConfig {
    if (features[0].length !== 2) {
      throw new Error("Decision boundary visualization requires 2D features");
    }

    // Create mesh for background
    const xMin = Math.min(...features.map((f) => f[0])) - 1;
    const xMax = Math.max(...features.map((f) => f[0])) + 1;
    const yMin = Math.min(...features.map((f) => f[1])) - 1;
    const yMax = Math.max(...features.map((f) => f[1])) + 1;

    const step = 0.1;
    const meshX: Vector = [];
    const meshY: Vector = [];

    for (let x = xMin; x <= xMax; x += step) {
      for (let y = yMin; y <= yMax; y += step) {
        meshX.push(x);
        meshY.push(y);
      }
    }

    const points: Point[] = features.map((point, i) => ({
      x: point[0],
      y: point[1],
      label: targets ? `True: ${targets[i]}, Pred: ${predictions[i]}` : undefined,
    }));

    return {
      type: "scatter",
      title,
      data: {
        points,
        dimensions: 2,
      } as ScatterData,
      axes: {
        x: this.createAxis([...meshX, ...features.map((f) => f[0])], "x₁"),
        y: this.createAxis([...meshY, ...features.map((f) => f[1])], "x₂"),
      },
      theme: this.defaultTheme,
    };
  }

  /**
   * Generate confusion matrix visualization.
   */
  confusionMatrix(
    trueLabels: Vector,
    predictions: Vector,
    numClasses = 2
  ): ChartConfig {
    const matrix: Matrix = Array(numClasses)
      .fill(0)
      .map(() => Array(numClasses).fill(0));

    for (let i = 0; i < trueLabels.length; i++) {
      const trueIdx = Math.floor(trueLabels[i]);
      const predIdx = Math.floor(predictions[i]);
      if (trueIdx < numClasses && predIdx < numClasses) {
        matrix[trueIdx][predIdx]++;
      }
    }

    return {
      type: "matrix",
      title: "Confusion Matrix",
      data: {
        matrix,
        labels: {
          rows: Array(numClasses)
            .fill(0)
            .map((_, i) => `True ${i}`),
          cols: Array(numClasses)
            .fill(0)
            .map((_, i) => `Pred ${i}`),
        },
      } as MatrixData,
      theme: this.defaultTheme,
    };
  }

  /**
   * Generate ROC curve (Receiver Operating Characteristic).
   */
  rocCurve(
    trueLabels: Vector,
    predictedProbs: Vector,
    title = "ROC Curve"
  ): ChartConfig {
    // Sort by predicted probability
    const indices = Array.from({ length: predictedProbs.length }, (_, i) => i).sort(
      (a, b) => predictedProbs[b] - predictedProbs[a]
    );

    const tpr: Vector = [];
    const fpr: Vector = [];

    let tp = 0;
    let fp = 0;
    const totalPos = trueLabels.reduce((a, b) => a + b, 0);
    const totalNeg = trueLabels.length - totalPos;

    tpr.push(0);
    fpr.push(0);

    for (const i of indices) {
      if (trueLabels[i] === 1) {
        tp++;
      } else {
        fp++;
      }
      tpr.push(tp / totalPos);
      fpr.push(fp / totalNeg);
    }

    return {
      type: "line",
      title,
      data: { x: fpr, y: tpr, label: "ROC" } as LineData,
      axes: {
        x: this.createAxis([0, 1], "FPR", "False Positive Rate"),
        y: this.createAxis([0, 1], "TPR", "True Positive Rate"),
      },
      theme: this.defaultTheme,
    };
  }

  /**
   * Generate loss curve visualization.
   */
  lossCurve(losses: Vector, title = "Training Loss"): ChartConfig {
    const iterations = Array.from({ length: losses.length }, (_, i) => i);

    return {
      type: "line",
      title,
      data: { x: iterations, y: losses, label: "Loss" } as LineData,
      axes: {
        x: this.createAxis(iterations, "Iteration", "Training Iteration"),
        y: this.createAxis(losses, "Loss", "Loss Value"),
      },
      theme: this.defaultTheme,
    };
  }

  /**
   * Generate accuracy curve visualization.
   */
  accuracyCurve(accuracies: Vector, title = "Training Accuracy"): ChartConfig {
    const iterations = Array.from(
      { length: accuracies.length },
      (_, i) => i
    );

    return {
      type: "line",
      title,
      data: { x: iterations, y: accuracies, label: "Accuracy" } as LineData,
      axes: {
        x: this.createAxis(iterations, "Iteration", "Training Iteration"),
        y: this.createAxis(accuracies, "Accuracy %", "Accuracy Percentage"),
      },
      theme: this.defaultTheme,
    };
  }

  /**
   * Create histogram configuration.
   */
  histogram(
    data: Vector,
    bins = 20,
    title = "Distribution"
  ): ChartConfig {
    const binEdges = this.createBins(data, bins);
    const binCounts = Array(bins).fill(0);

    for (const value of data) {
      for (let i = 0; i < bins; i++) {
        if (value >= binEdges[i] && value < binEdges[i + 1]) {
          binCounts[i]++;
          break;
        }
      }
    }

    const binCenters = binEdges.slice(0, -1).map(
      (edge, i) => (edge + binEdges[i + 1]) / 2
    );

    return {
      type: "histogram",
      title,
      data: { x: binCenters, y: binCounts } as LineData,
      axes: {
        x: this.createAxis(binCenters, "Value", "Bin Center"),
        y: this.createAxis(binCounts, "Frequency", "Count"),
      },
      theme: this.defaultTheme,
    };
  }

  /**
   * Normalize data to [0, 1] range.
   */
  normalize(data: Vector): Vector {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min;
    return range === 0 ? data.map(() => 0) : data.map((v) => (v - min) / range);
  }

  /**
   * Generate evenly-spaced bins for histogram.
   */
  private createBins(data: Vector, numBins: number): Vector {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / numBins;
    const bins: Vector = [];

    for (let i = 0; i <= numBins; i++) {
      bins.push(min + i * binWidth);
    }

    return bins;
  }

  /**
   * Create axis configuration from data.
   */
  private createAxis(data: Vector, label: string, description = ""): AxisConfig {
    const values = data.filter((v) => !isNaN(v) && isFinite(v));
    if (values.length === 0) {
      return { label, min: 0, max: 1 };
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const padding = (max - min) * 0.1;

    return {
      label: description || label,
      min: min - padding,
      max: max + padding,
      scale: "linear",
    };
  }

  /**
   * Scale data to canvas size.
   */
  scaleToCanvas(
    data: Vector,
    fromMin: number,
    fromMax: number,
    toMin: number,
    toMax: number
  ): Vector {
    const fromRange = fromMax - fromMin;
    const toRange = toMax - toMin;

    return data.map((v) => {
      const normalized = (v - fromMin) / fromRange;
      return toMin + normalized * toRange;
    });
  }

  /**
   * Compute color for value in gradient.
   */
  getColor(value: number, min: number, max: number): string {
    const normalized = (value - min) / (max - min);
    const hue = ((1 - normalized) * 240) / 360; // Blue to Red

    return `hsl(${hue * 360}, 100%, 50%)`;
  }

  /**
   * Create gradient palette.
   */
  createPalette(count: number, startColor = "#3b82f6", endColor = "#ef4444"): string[] {
    const palette: string[] = [];
    for (let i = 0; i < count; i++) {
      const t = i / (count - 1);
      palette.push(this.interpolateColor(startColor, endColor, t));
    }
    return palette;
  }

  /**
   * Interpolate between two hex colors.
   */
  private interpolateColor(
    color1: string,
    color2: string,
    t: number
  ): string {
    const r1 = parseInt(color1.slice(1, 3), 16);
    const g1 = parseInt(color1.slice(3, 5), 16);
    const b1 = parseInt(color1.slice(5, 7), 16);

    const r2 = parseInt(color2.slice(1, 3), 16);
    const g2 = parseInt(color2.slice(3, 5), 16);
    const b2 = parseInt(color2.slice(5, 7), 16);

    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);

    return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
  }
}
