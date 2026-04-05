"use client";

import React from "react";
import { ChartConfig } from "@/core/visualization-engine";

/**
 * Visualization Panel: Displays charts and plots
 */

interface VisualizationPanelProps {
  chart?: ChartConfig;
  isLoading?: boolean;
}

export function VisualizationPanel({
  chart,
  isLoading = false,
}: VisualizationPanelProps) {
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
        <p className="text-foreground/60">Loading visualization...</p>
      </div>
    );
  }

  if (!chart) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-foreground/60">
        <p className="text-lg font-semibold mb-2">📊 No Data Yet</p>
        <p className="text-sm">Configure and train an algorithm to see visualization</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col">
      <h3 className="text-lg font-semibold mb-4">{chart.title}</h3>

      {/* Placeholder for actual chart rendering */}
      {/* In production, this would render with Plotly or D3 */}
      <div className="flex-1 flex items-center justify-center bg-muted rounded-lg border-2 border-dashed border-border">
        <div className="text-center">
          <p className="text-sm text-foreground/60 mb-2">
            Chart Type: <span className="font-mono text-primary">{chart.type}</span>
          </p>
          <p className="text-xs text-foreground/40">
            Rendering engine coming in Phase 3
          </p>
          <div className="mt-4 p-3 bg-background rounded text-left text-xs max-w-xs">
            <p className="font-mono text-primary mb-2">Configuration:</p>
            <pre className="text-foreground/70 overflow-auto max-h-32">
              {JSON.stringify(
                { type: chart.type, title: chart.title, axes: chart.axes },
                null,
                2
              )}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
