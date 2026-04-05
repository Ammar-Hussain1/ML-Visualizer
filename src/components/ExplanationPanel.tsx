"use client";

import React from "react";
import { Step } from "@/core/types";
import { ExplanationEngine } from "@/core/explanation-engine";

/**
 * Explanation Panel: Displays mathematical steps
 */

interface ExplanationPanelProps {
  steps?: Step[];
  currentStepIndex?: number;
  isDarkMode?: boolean;
}

export function ExplanationPanel({
  steps = [],
  currentStepIndex = -1,
  isDarkMode = true,
}: ExplanationPanelProps) {
  const engine = new ExplanationEngine();

  if (steps.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-foreground/60">
        <p className="text-lg font-semibold mb-2">📐 Mathematics</p>
        <p className="text-xs">Steps will appear here</p>
      </div>
    );
  }

  const currentStep = currentStepIndex >= 0 ? steps[currentStepIndex] : null;

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Current Step */}
      {currentStep && (
        <div className="flex-1 overflow-auto space-y-3">
          <div>
            <h3 className="font-bold text-primary mb-2">
              Step {currentStepIndex + 1} of {steps.length}
            </h3>
            <p className="text-sm font-semibold text-foreground mb-2">
              {currentStep.label}
            </p>
          </div>

          {/* Formula */}
          <div className="bg-background rounded p-3 border border-border">
            <p className="text-xs font-mono text-foreground/70 mb-1">Formula</p>
            <code className="text-sm text-primary font-mono break-words">
              {currentStep.formula}
            </code>
          </div>

          {/* Substitution */}
          <div className="bg-background rounded p-3 border border-border">
            <p className="text-xs font-mono text-foreground/70 mb-1">
              Substitution
            </p>
            <code className="text-sm text-accent font-mono break-words">
              {currentStep.substitution}
            </code>
          </div>

          {/* Result */}
          <div className="bg-background rounded p-3 border-2 border-primary">
            <p className="text-xs font-mono text-foreground/70 mb-1">Result</p>
            <code className="text-sm text-primary font-bold font-mono">
              {currentStep.result}
            </code>
          </div>

          {/* Description */}
          {currentStep.description && (
            <div className="bg-background rounded p-3 border border-border/50">
              <p className="text-xs text-foreground/70">
                {currentStep.description}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Step Navigation */}
      <div className="border-t border-border pt-4 space-y-2">
        <p className="text-xs font-semibold text-foreground/60">
          Steps Completed: {currentStepIndex + 1}
        </p>

        {/* Step List */}
        <div className="max-h-32 overflow-y-auto space-y-1">
          {steps.slice(0, 5).map((step, idx) => (
            <div
              key={step.id}
              className={`text-xs p-2 rounded border ${
                idx === currentStepIndex
                  ? "bg-primary/10 border-primary text-primary font-semibold"
                  : idx < currentStepIndex
                    ? "bg-muted border-border/50 text-foreground/70 line-through"
                    : "bg-background border-border/30 text-foreground/50"
              }`}
            >
              {idx + 1}. {step.label}
            </div>
          ))}
          {steps.length > 5 && (
            <p className="text-xs text-foreground/40 px-2">
              ... and {steps.length - 5} more steps
            </p>
          )}
        </div>
      </div>

      {/* Info */}
      <div className="text-xs text-foreground/50 p-2 bg-background rounded border border-border/30">
        <p>💡 Every computation is shown step-by-step</p>
      </div>
    </div>
  );
}
