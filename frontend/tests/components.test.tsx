import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { EmptyState, SimulatedBanner, StatTile, seriesColor } from "../src/components/ui";
import { DecisionView } from "../src/pages/Routing";

describe("ui components", () => {
  it("shows the simulated-data banner only when asked", () => {
    const { rerender } = render(<SimulatedBanner show={false} />);
    expect(screen.queryByRole("status")).toBeNull();
    rerender(<SimulatedBanner show />);
    expect(screen.getByRole("status")).toHaveTextContent("Simulated data");
  });

  it("renders stat tiles and empty states", () => {
    render(
      <>
        <StatTile label="Success rate" value="99.9%" hint="3 recovered" />
        <EmptyState title="Nothing here" />
      </>,
    );
    expect(screen.getByText("Success rate")).toBeInTheDocument();
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });

  it("keeps series colours stable per entity, not per rank", () => {
    const order = ["a", "b", "c"];
    expect(seriesColor("b", order)).toBe(seriesColor("b", ["a", "b", "c"]));
    expect(seriesColor("zzz", order)).toBe("#9a9893");
  });

  it("explains a routing decision", () => {
    render(
      <DecisionView
        d={{
          strategy: "learned", selected_model: "haiku", fallback_chain: ["sonnet"], reason: "cheapest_meeting_constraints",
          confidence: 0.91, predicted_task: "code", quality_threshold: 0.8, latency_budget_ms: null, uncertain: false, decision_ms: 0.4,
          task_probs: { code: 0.9, reasoning: 0.1 },
          candidates: [
            { model_id: "haiku", eligible: true, ineligible_reason: null, p_quality: 0.91, p_quality_adjusted: 0.86, expected_output_tokens: 100, expected_cost_usd: 0.0001, expected_latency_ms: 900, meets_quality: true, meets_latency: true },
            { model_id: "sonnet", eligible: true, ineligible_reason: null, p_quality: 0.95, p_quality_adjusted: 0.93, expected_output_tokens: 120, expected_cost_usd: 0.002, expected_latency_ms: 1500, meets_quality: true, meets_latency: true },
          ],
        }}
      />,
    );
    expect(screen.getByText("cheapest_meeting_constraints")).toBeInTheDocument();
    expect(screen.getAllByText("haiku").length).toBeGreaterThan(0);
    expect(screen.getAllByText("sonnet").length).toBeGreaterThan(0);
  });
});
