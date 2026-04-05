# ML Visual Learning Platform (Production-Grade, Math-Focused)

## 1. Vision

A production-grade, math-first ML platform where every algorithm is:

* Derived mathematically
* Executed step-by-step
* Visualized interactively

No black boxes. Every number must be traceable.

---

## 2. Core Philosophy (STRICT)

Every concept must include:

1. Formal definition
2. Mathematical derivation
3. Step-by-step numeric computation
4. Visual representation
5. Interactive manipulation

If a feature cannot show math → it is incomplete.

---

## 3. System Architecture (Production Grade)

### 3.1 High-Level Architecture

```ts
/apps/web (Next.js)
/core (math engine)
/visual (rendering engine)
/engine (step system)
/components (UI)
/modules (algorithms)
```

---

## 4. Universal Algorithm Interface (CRITICAL)

All ML algorithms must implement:

```ts
interface Algorithm {
  name: string

  initialize(config): State

  forward(state): Step[]

  backward?(state): Step[]

  update?(state): Step[]
}
```

This ensures:

* Plug-and-play modules
* Consistent visualization
* Reusable UI

---

## 5. Step Engine (CORE SYSTEM)

Everything runs through a deterministic step engine.

```ts
interface Step {
  id: string
  label: string
  formula: string
  substitution: string
  result: string
  execute(): void
}
```

### Pipeline

1. Generate steps
2. Execute step
3. Update state
4. Trigger visualization
5. Render math explanation

---

## 6. Math Engine (CORE)

### Structure

```ts
/core
  /probability
  /linear_algebra
  /optimization
  /statistics
```

### Rules

* No external ML libraries
* Deterministic outputs
* Symbolic + numeric representation

---

## 7. Visualization Engine

### Visual Primitives (Reusable)

* Graph (Neural Networks)
* Scatter Plot (Clustering, Regression)
* Line/Curve (Loss, sigmoid)
* Matrix (Confusion Matrix)
* Tree (Decision Trees)
* Surface (Optimization landscape)

---

## 8. Explanation Engine (MOST IMPORTANT)

Each step renders:

```txt
Formula:
z = w1*x1 + w2*x2 + b

Substitution:
= (0.2)(1) + (-0.3)(0) + 0.1

Result:
= 0.3
```

### Requirements

* Always show symbols
* Always show numbers
* Always show final value

---

## 9. UI Architecture

### Layout

* Left: Controls (inputs, hyperparameters)
* Center: Visualization
* Right: Math Panel (formulas + steps)

---

## 10. Theme System (STRICT IMPLEMENTATION)

Use provided CSS variables as design tokens.

### Rules

* NO hardcoded colors
* ONLY use CSS variables
* Tailwind mapped to variables

### Example

```ts
bg-background
text-foreground
border-border
```

### Visual Priority

* Math readability > aesthetics
* High contrast for formulas
* Highlight active computations using --primary

---

## 11. Modules (Curriculum-Aligned)

### 11.1 Foundations

* Feature vectors
* Data pipelines

---

### 11.2 Naïve Bayes

Math Focus:

* P(y|x) = P(x|y)P(y)/P(x)

Visuals:

* Probability tables
* Likelihood computation

---

### 11.3 Supervised Learning Setup

Visuals:

* Train/test split
* Overfitting curves

---

### 11.4 Evaluation Metrics

Visuals:

* Confusion matrix
* ROC / PR curves

---

### 11.5 Linear Regression

Math:

* Cost function
* Gradient descent

Visuals:

* Line fitting
* Loss curve

---

### 11.6 Logistic Regression

Math:

* Sigmoid
* Cross-entropy

Visuals:

* Decision boundary

---

### 11.7 Neural Networks

Math:

* Chain rule
* Backpropagation

Visuals:

* Graph + gradient flow

---

### 11.8 KNN

Math:

* Distance metrics

Visuals:

* Neighbor regions

---

### 11.9 Decision Trees + Ensembles

Math:

* Entropy, Gini

Visuals:

* Tree splitting

---

### 11.10 Clustering

Math:

* K-means objective

Visuals:

* Cluster updates

---

### 11.11 PCA

Math:

* Eigenvectors

Visuals:

* Projection

---

### 11.12 SVM

Math:

* Margin maximization
* Hinge loss

Visuals:

* Hyperplane

---

## 12. State Management

Use Zustand

```ts
store = {
  currentAlgorithm,
  state,
  steps,
  currentStepIndex
}
```

---

## 13. Performance

* Dynamic imports for heavy modules
* Memoized computations
* Limit dataset size

---

## 14. Development Phases

### Phase 1

* Step engine + math engine

### Phase 2

* Linear + Logistic Regression

### Phase 3

* Neural Networks

### Phase 4

* Trees + KNN

### Phase 5

* Clustering + PCA + SVM

---

## 15. Non-Negotiables

* Every number must be explainable
* Every step must be visible
* No hidden computation
* Math correctness over UI speed

---

## 16. Final Outcome

A production-grade ML platform where:

* Users learn from derivations
* Users see every computation
* Algorithms become transparent

---

End of Plan
