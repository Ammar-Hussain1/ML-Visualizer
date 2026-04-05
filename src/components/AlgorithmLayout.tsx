"use client";

import React from "react";

/**
 * Main layout component with three panels:
 * - Left: Control Panel
 * - Center: Visualization
 * - Right: Explanation Panel
 */

interface LayoutProps {
  controlPanel: React.ReactNode;
  visualization: React.ReactNode;
  explanation: React.ReactNode;
}

export function AlgorithmLayout({
  controlPanel,
  visualization,
  explanation,
}: LayoutProps) {
  return (
    <div className="flex h-screen w-full bg-background text-foreground">
      {/* Left Panel: Controls */}
      <div className="w-1/4 border-r border-border bg-muted p-6 overflow-y-auto">
        <div className="space-y-6">
          <h2 className="text-xl font-bold">Configuration</h2>
          {controlPanel}
        </div>
      </div>

      {/* Center Panel: Visualization */}
      <div className="flex-1 border-r border-border p-6 overflow-auto bg-background">
        <div className="h-full flex flex-col">
          <h2 className="text-xl font-bold mb-4">Visualization</h2>
          <div className="flex-1 flex items-center justify-center bg-card rounded-lg border border-border">
            {visualization}
          </div>
        </div>
      </div>

      {/* Right Panel: Explanation */}
      <div className="w-1/3 border-l border-border bg-muted p-6 overflow-y-auto">
        <div className="space-y-4">
          <h2 className="text-xl font-bold">Mathematics</h2>
          <div className="bg-card rounded-lg border border-border p-4">
            {explanation}
          </div>
        </div>
      </div>
    </div>
  );
}
