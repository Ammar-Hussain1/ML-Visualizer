# ML Visualizer - Production-Grade Math-First Machine Learning Platform

A comprehensive, interactive machine learning platform where **every algorithm is derived mathematically, executed step-by-step, and visualized interactively**. No black boxes—every computation is traceable from formula to result.

## 🎯 Philosophy

This platform embodies a strict principle: **If a feature cannot show the math, it is incomplete.**

Every algorithm includes:
1. **Formal mathematical definition**
2. **Step-by-step numeric computation**
3. **Visual representation**
4. **Interactive step-by-step walkthrough**
5. **Explanation panel with formulas, substitutions, and results**

## ✨ Implemented Algorithms

### Supervised Learning
- **Linear Regression** - Gradient descent optimization with loss visualization
- **Logistic Regression** - Binary/multiclass classification with sigmoid and cross-entropy
- **Neural Networks** - Multi-layer perceptron with backpropagation, layer visualization, and gradient flow

### Instance-Based Learning
- **K-Nearest Neighbors (KNN)** - Lazy learning with 5 distance metrics (Euclidean, Manhattan, Minkowski, Cosine, Hamming) and interactive decision region visualization

### Tree-Based Learning
- **Decision Trees** - Recursive splitting with Gini impurity, information gain, and tree decision visualization

### Unsupervised Learning
- **K-Means Clustering** - Centroid-based clustering with Voronoi-like decision regions, centroid tracking, and inertia minimization
- **Principal Component Analysis (PCA)** - Dimensionality reduction with eigenvector computation, variance explained tracking, and 2D/3D projection visualization

## 🏗️ Architecture

### Core Systems
```
/core
  ├── Step Engine - Deterministic execution with undo/redo
  ├── Training Engine - Orchestrates algorithm execution with callbacks
  ├── Explanation Engine - Formats mathematical steps for display
  ├── Visualization Engine - Generates chart configurations
  ├── Math Engine - All algorithms from scratch (no external ML libraries)
  └── State Management - Zustand for global state
```

### UI Components
```
/components
  ├── AlgorithmLayout - 3-panel responsive grid
  ├── ControlPanel - Hyperparameter tuning
  ├── VisualizationPanel - Chart display
  ├── ExplanationPanel - Step-by-step math breakdown
  └── Algorithm-specific visualizations (NeuralNetwork, KNN, DecisionTree, KMeans, PCA)
```

### Algorithm Interface (Universal)
All algorithms implement a consistent interface:
```typescript
interface Algorithm<TState, TResult> {
  name: string;
  config: AlgorithmConfig;
  initialize(features, targets): TState;
  forward(state, input): { result, steps };
  update(state): { state, steps };
  isReached(state): boolean;
}
```

## 🚀 Getting Started

### Installation & Development
```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Open browser at http://localhost:3000
```

### Build for Production
```bash
npm run build
npm run start
```

## 💻 Tech Stack

- **Framework**: Next.js 16.2.2 with Turbopack
- **UI**: React 19, TypeScript, Tailwind CSS
- **State**: Zustand (lightweight global store)
- **Math**: Pure TypeScript (no external ML libraries)
- **Styling**: CSS variables (theme system with light/dark mode)

## 📊 How It Works

### The Step Engine Pipeline
1. **Generate Steps** - Algorithm produces detailed computation steps
2. **Execute Step** - User navigates through each step with back/forward
3. **Update State** - Each step updates the algorithm state
4. **Trigger Visualization** - Visualization components re-render
5. **Render Explanation** - Formula, substitution, and result displayed

### Example: Linear Regression
```
Formula: ŷ = w·x + b
Substitution: = (0.2)(1) + 0.1
Result: = 0.3
```

Each parameter change, gradient computation, and weight update is a traceable step.

## 🎨 Features

### Visualizations
- **Scatter plots** - Regression, classification, clustering
- **Network graphs** - Neural network architecture with layer connections
- **Decision regions** - KNN, Decision Trees with Voronoi boundaries
- **Loss curves** - Training progress and convergence
- **Principal components** - PCA projection with variance explained
- **Cluster assignments** - K-Means with centroid positions

### Interactive Controls
- **Hyperparameter sliders** - Adjust learning rate, iterations, K, max depth
- **Algorithm selector** - Choose from 7 algorithms
- **Dataset size control** - Generate 50-200 samples
- **Step-by-step navigation** - Rewind, play, skip through computation

### Mathematical Rigor
- All distance metrics implemented from scratch (Euclidean, Manhattan, Cosine, Hamming, Minkowski)
- Activation functions with derivatives (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions: MSE, Cross-Entropy, MAE with L1/L2 regularization
- Tree utilities: Entropy, Gini impurity, Information gain
- Dimensionality reduction: Power iteration for eigendecomposition

## 📁 Project Structure

```
/src
  /app - Next.js app directory
  /components - React UI components
  /core
    /algorithms - Algorithm implementations
    /math - Math utilities and operations
    *.ts - Engines (Step, Training, Explanation, Visualization)
  /public - Static assets
```

## 🧪 Testing

The project builds successfully with zero TypeScript errors:
```bash
✓ Compiled successfully in 3.1s
✓ Finished TypeScript in 2.2s
✓ Collecting page data...
✓ Generating static pages...
```

All algorithms:
- ✅ Generate step-by-step explanations
- ✅ Track state deterministically
- ✅ Visualize results interactively
- ✅ Handle edge cases gracefully

## 📚 Curriculum Alignment

This platform covers fundamental ML concepts in order:
1. **Foundations** - Linear algebra, data preprocessing
2. **Supervised Learning** - Regression, classification
3. **Neural Networks** - Deep learning, backpropagation
4. **Instance-Based** - Lazy learning, similarity metrics
5. **Tree-Based** - Recursive splitting, entropy
6. **Unsupervised** - Clustering, dimensionality reduction

## Future Enhancements

Planned modules for Phase 5+:
- Naive Bayes (probabilistic classification)
- Support Vector Machines (margin maximization)
- Evaluation metrics (confusion matrix, ROC/PR curves)
- Ensemble methods (bagging, boosting)
- Advanced clustering (DBSCAN, hierarchical)

## 📄 License

This project is open source and available under the MIT License.

## 🎓 Educational Use

This platform is designed for:
- **Students** learning ML theory and implementation
- **Educators** demonstrating algorithms with mathematical rigor
- **Practitioners** validating algorithm behavior step-by-step
- **Researchers** exploring new algorithm visualizations

Every line of code is deterministic, traceable, and mathematically sound.

---

**Built with precision. Visualized with clarity. Explaining with mathematics.**
