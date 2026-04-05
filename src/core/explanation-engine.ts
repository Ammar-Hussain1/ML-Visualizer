import { Step } from "./types";

/**
 * Explanation Engine: Renders mathematical steps with full transparency.
 * 
 * Responsibilities:
 * - Format steps for display
 * - Handle LaTeX rendering
 * - Provide structured step information
 * - Support highlighting and emphasis
 * 
 * Philosophy: Every step must be human-readable and mathematically clear
 */

export interface FormattedStep {
  id: string;
  label: string;
  formula: {
    latex: string;
    description: string;
  };
  substitution: {
    text: string;
    highlighted: string[];
  };
  result: {
    value: string;
    numeric: number | null;
  };
  context: string;
  explanation: Record<string, string>;
}

export class ExplanationEngine {
  /**
   * Format a step for display.
   */
  formatStep(step: Step): FormattedStep {
    return {
      id: step.id,
      label: step.label,
      formula: {
        latex: this.latexify(step.formula),
        description: step.description || "See formula above",
      },
      substitution: {
        text: step.substitution,
        highlighted: this.extractNumbers(step.substitution),
      },
      result: {
        value: step.result,
        numeric: this.extractNumeric(step.result),
      },
      context: this.generateContext(step),
      explanation: this.breakdown(step),
    };
  }

  /**
   * Convert mathematical notation to LaTeX format.
   */
  private latexify(formula: string): string {
    // Already LaTeX-like, just clean it up
    return formula
      .replace(/'/g, "′") // prime symbol
      .replace(/α/g, "\\alpha")
      .replace(/β/g, "\\beta")
      .replace(/γ/g, "\\gamma")
      .replace(/λ/g, "\\lambda")
      .replace(/σ/g, "\\sigma")
      .replace(/Σ/g, "\\Sigma")
      .replace(/∂/g, "\\partial")
      .replace(/∞/g, "\\infty");
  }

  /**
   * Extract numeric values from substitution text.
   */
  private extractNumbers(text: string): string[] {
    const numberRegex = /-?\d+\.?\d*([eE][+-]?\d+)?/g;
    return (text.match(numberRegex) || []).filter(
      (n) => !Number.isNaN(parseFloat(n))
    );
  }

  /**
   * Extract the final numeric result from result text.
   */
  private extractNumeric(text: string): number | null {
    const match = text.match(/-?\d+\.?\d*([eE][+-]?\d+)?(?![\d.])/);
    return match ? parseFloat(match[0]) : null;
  }

  /**
   * Generate human-readable context for a step.
   */
  private generateContext(step: Step): string {
    if (step.description) {
      return step.description;
    }

    // Default context based on label
    const contexts: Record<string, string> = {
      "forward pass": "Generate predictions for all training samples",
      "compute loss":
        "Calculate how far predictions are from actual targets",
      "compute gradients":
        "Determine direction and magnitude of parameter updates",
      "update parameters":
        "Move weights and bias towards minimizing loss",
      "apply activation":
        "Transform raw output into appropriate range",
      "decision boundary":
        "Determine class prediction based on threshold",
    };

    for (const [key, value] of Object.entries(contexts)) {
      if (step.label.toLowerCase().includes(key)) {
        return value;
      }
    }

    return "Mathematical operation";
  }

  /**
   * Break down a step into human-readable parts.
   */
  private breakdown(step: Step): Record<string, string> {
    return {
      what: step.label,
      why: this.generateContext(step),
      formula: step.formula,
      numbers: step.substitution,
      answer: step.result,
    };
  }

  /**
   * Get step summary for visualization.
   */
  getStepSummary(step: Step): string {
    const lines = [
      `📊 ${step.label}`,
      `   Formula: ${step.formula}`,
      `   = ${step.substitution}`,
      `   = ${step.result}`,
    ];
    if (step.description) {
      lines.push(`   Note: ${step.description}`);
    }
    return lines.join("\n");
  }

  /**
   * Get formatted step group (multiple related steps).
   */
  getStepGroupSummary(steps: Step[], groupLabel: string): string {
    const lines = [
      `🔍 ${groupLabel}`,
      `   Total steps: ${steps.length}`,
      `   Steps:`,
    ];

    steps.forEach((step, i) => {
      lines.push(`     ${i + 1}. ${step.label}`);
    });

    return lines.join("\n");
  }

  /**
   * Validate step contains all required fields.
   */
  validateStep(step: Step): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!step.id) errors.push("Missing step ID");
    if (!step.label) errors.push("Missing step label");
    if (!step.formula) errors.push("Missing step formula");
    if (!step.substitution) errors.push("Missing step substitution");
    if (!step.result) errors.push("Missing step result");

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Format steps for console output (debugging/demo).
   */
  formatForConsole(step: Step): string {
    return `
    ┌─────────────────────────────────────────
    │ ${step.label}
    ├─────────────────────────────────────────
    │ Formula:       ${step.formula}
    │ Substitution:  ${step.substitution}
    │ Result:        ${step.result}
    ${step.description ? `│ Note:          ${step.description}\n` : ""}    └─────────────────────────────────────────
    `;
  }

  /**
   * Create HTML representation of a step (for web display).
   */
  toHTML(step: Step): string {
    const numericResult = this.extractNumeric(step.result);

    return `
      <div class="step" data-step-id="${step.id}">
        <h3 class="step-label">${this.htmlEscape(step.label)}</h3>
        <div class="step-formula">
          <strong>Formula:</strong>
          <code>${this.htmlEscape(step.formula)}</code>
        </div>
        <div class="step-substitution">
          <strong>Substitution:</strong>
          <code>${this.htmlEscape(step.substitution)}</code>
        </div>
        <div class="step-result">
          <strong>Result:</strong>
          <code>${this.htmlEscape(step.result)}</code>
          ${numericResult !== null ? `<span class="numeric">${numericResult}</span>` : ""}
        </div>
        ${step.description ? `<p class="step-description">${this.htmlEscape(step.description)}</p>` : ""}
      </div>
    `;
  }

  /**
   * Helper: escape HTML special characters.
   */
  private htmlEscape(text: string): string {
    const map: Record<string, string> = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return text.replace(/[&<>"']/g, (char) => map[char]);
  }

  /**
   * Create a comparison between two consecutive steps.
   */
  compareSteps(before: Step, after: Step): string {
    return `
    Before: ${before.label}
    Result: ${before.result}
    
    After:  ${after.label}
    Result: ${after.result}
    `;
  }

  /**
   * Generate learning notes from steps.
   */
  generateNotes(steps: Step[]): string {
    const notes: string[] = [
      "# Mathematical Computation Notes\n",
    ];

    let currentGroup = "";
    let groupSteps: Step[] = [];

    steps.forEach((step, i) => {
      const stepType = step.label.split(" ")[0];

      if (currentGroup !== stepType) {
        if (groupSteps.length > 0) {
          notes.push(`\n## ${currentGroup} (${groupSteps.length} steps)\n`);
          groupSteps.forEach((s) => {
            notes.push(`- **${s.label}**: ${s.result}`);
          });
        }
        currentGroup = stepType;
        groupSteps = [];
      }

      groupSteps.push(step);
    });

    // Final group
    if (groupSteps.length > 0) {
      notes.push(`\n## ${currentGroup} (${groupSteps.length} steps)\n`);
      groupSteps.forEach((s) => {
        notes.push(`- **${s.label}**: ${s.result}`);
      });
    }

    return notes.join("\n");
  }
}
