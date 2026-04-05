/**
 * Phase 1 Example: Linear Regression with Full Step Visualization
 * 
 * This example demonstrates:
 * 1. Creating synthetic training data
 * 2. Initializing the LinearRegression algorithm
 * 3. Running the training engine with step callbacks
 * 4. Observing every mathematical step
 * 
 * Run with: npx ts-node src/core/examples/phase1-demo.ts
 */

import { LinearRegression } from "../algorithms/linear-regression";
import { TrainingEngine } from "../training-engine";
import { Matrix, Vector } from "../types";

/**
 * Generate simple linear regression data: y = 2x + 1 + noise
 */
function generateTrainingData(
  samples: number = 10
): { features: Matrix; targets: Vector } {
  const features: Matrix = [];
  const targets: Vector = [];

  for (let i = 0; i < samples; i++) {
    const x = Math.random() * 10 - 5; // Random x in [-5, 5]
    const noise = (Math.random() - 0.5) * 0.5;
    const y = 2 * x + 1 + noise; // True relationship: y = 2x + 1

    features.push([x]);
    targets.push(y);
  }

  return { features, targets };
}

/**
 * Main demo function
 */
async function runDemo() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  ML VISUALIZER - PHASE 1: Step Engine & Math Engine Demo");
  console.log("═══════════════════════════════════════════════════════════\n");

  // Step 1: Generate training data
  console.log("1. GENERATING TRAINING DATA");
  console.log("   Target: y = 2x + 1");
  const { features, targets } = generateTrainingData(5);
  console.log(`   Generated ${features.length} samples\n`);

  // Step 2: Create algorithm
  console.log("2. INITIALIZING LINEAR REGRESSION ALGORITHM");
  const algorithm = new LinearRegression({
    learningRate: 0.1,
    maxIterations: 5,
  });
  console.log(`   Algorithm: ${algorithm.name}`);
  console.log(`   Learning Rate: 0.1`);
  console.log(`   Max Iterations: 5\n`);

  // Step 3: Create training engine with callbacks
  console.log("3. SETTING UP TRAINING ENGINE\n");
  let stepCount = 0;
  const engine = new TrainingEngine(algorithm, {
    onStepComplete: (step, progress) => {
      stepCount++;
      console.log(`   [Step ${stepCount}] ${step.label}`);
      console.log(`   Formula:      ${step.formula}`);
      console.log(`   Substitution: ${step.substitution}`);
      console.log(`   Result:       ${step.result}\n`);
    },
    onIterationComplete: (progress) => {
      console.log(
        `   ──── Iteration ${progress.iteration}/${progress.totalIterations} Complete ────\n`
      );
    },
    onTrainingComplete: () => {
      console.log("   ✓ Training Complete!\n");
    },
  });

  // Step 4: Initialize
  console.log("4. INITIALIZING ALGORITHM STATE\n");
  engine.initialize(features, targets);

  // Step 5: Run training
  console.log("5. RUNNING TRAINING LOOP\n");
  await engine.train(5, true);

  // Step 6: Display results
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  TRAINING RESULTS");
  console.log("═══════════════════════════════════════════════════════════\n");

  const allSteps = engine.getSteps();
  console.log(`Total Steps Executed: ${allSteps.length}`);
  console.log(`Total Iterations: 5\n`);

  console.log("Sample of Executed Steps:");
  allSteps.slice(0, 10).forEach((step, i) => {
    console.log(`  ${i + 1}. ${step.label}`);
  });
  if (allSteps.length > 10) {
    console.log(`  ... and ${allSteps.length - 10} more steps\n`);
  }

  console.log("═══════════════════════════════════════════════════════════");
  console.log("  PHASE 1 COMPLETE: All Systems Operational");
  console.log("═══════════════════════════════════════════════════════════");
}

// Run the demo
runDemo().catch(console.error);
