import { describe, expect, it } from "vitest";

import { fmt } from "../src/services/api";

describe("formatters", () => {
  it("formats percentages and missing values", () => {
    expect(fmt.pct(0.8723)).toBe("87.2%");
    expect(fmt.pct(null)).toBe("—");
  });
  it("formats latency with sensible units", () => {
    expect(fmt.ms(4.25)).toBe("4.25 ms");
    expect(fmt.ms(950)).toBe("950 ms");
    expect(fmt.ms(1530)).toBe("1.53 s");
  });
  it("formats tiny dollar amounts without rounding them to zero", () => {
    expect(fmt.usd(0)).toBe("$0");
    expect(fmt.usd(0.000123)).toBe("$0.00012");
    expect(fmt.usd(12.5)).toBe("$12.50");
  });
});
