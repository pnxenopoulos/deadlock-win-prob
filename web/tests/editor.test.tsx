import { afterEach, expect, test } from "bun:test";
import { Window } from "happy-dom";
import { act } from "react";
import type { RawGraph } from "../src/ml/types";

const browser = new Window({ url: "http://localhost/" });
for (const key of [
  "window",
  "document",
  "navigator",
  "HTMLElement",
  "HTMLInputElement",
  "HTMLSelectElement",
  "Element",
  "Node",
  "Event",
  "MouseEvent",
  "KeyboardEvent",
  "FocusEvent",
  "localStorage",
  "MutationObserver",
]) {
  Object.defineProperty(globalThis, key, {
    value: key === "window" ? browser : browser[key as keyof Window],
    writable: true,
    configurable: true,
  });
}
Object.defineProperty(globalThis, "IS_REACT_ACT_ENVIRONMENT", {
  value: true,
  configurable: true,
});
process.env.BASE_URL = "/";
let request: { id: number; graphs: RawGraph[] };
class WorkerStub {
  onmessage?: (e: { data: unknown }) => void;
  onerror?: () => void;
  stopped = false;
  constructor(_url: URL, _options: unknown) {}
  postMessage(value: typeof request) {
    request = value;
    if (!this.stopped)
      this.onmessage?.({
        data: {
          id: value.id,
          probabilities: value.graphs.map(() => 0.6),
          modelVersion: "test-only-worker-stub",
        },
      });
  }
  terminate() {
    this.stopped = true;
  }
}
Object.defineProperty(globalThis, "Worker", {
  value: WorkerStub,
  configurable: true,
});
const { createRoot } = await import("react-dom/client");
const { App } = await import("../src/App");
let root: ReturnType<typeof createRoot> | undefined;
async function flush() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 220));
  });
}
async function mount() {
  localStorage.clear();
  document.body.innerHTML = '<div id="test-root"></div>';
  root = createRoot(document.getElementById("test-root")!);
  await act(() => root!.render(<App />));
  await flush();
}
afterEach(async () => {
  if (root) await act(() => root!.unmount());
  root = undefined;
  await browser.happyDOM.cancelAsync();
});
function click(element: Element | null) {
  if (!element) throw new Error("Missing control");
  return act(() =>
    element.dispatchEvent(new MouseEvent("click", { bubbles: true })),
  );
}
async function select(label: string, value: string) {
  const element = (document.querySelector(`select[aria-label="${label}"]`) ??
    Array.from(document.querySelectorAll("label.field"))
      .find((e) => e.querySelector("span")?.textContent === label)
      ?.querySelector("select")) as HTMLSelectElement;
  expect(element).not.toBeNull();
  await act(() => {
    element.value = value;
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
}
async function numeric(label: string, value: string) {
  const input = document.querySelector(
    `input[aria-label="${label}"]`,
  ) as HTMLInputElement;
  expect(input).not.toBeNull();
  await act(() => {
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )!.set!.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await act(() => {
    input.dispatchEvent(new FocusEvent("focusout", { bubbles: true }));
  });
}

test("the editor renders a full roster and sends edits to the prediction worker", async () => {
  await mount();
  expect(document.querySelectorAll(".hero-row")).toHaveLength(12);
  expect(document.querySelector(".probabilities")!.textContent).toContain(
    "60.0",
  );
  const before = request.graphs[0].players[0].x as number,
    z = request.graphs[0].players[0].z;
  const hero = document.querySelector(".map-hero")!;
  await act(() =>
    hero.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }),
    ),
  );
  await flush();
  expect(request.graphs[0].players[0].x).toBe(Math.round(before + 100));
  expect(request.graphs[0].players[0].z).toBe(z);
  await numeric("Health", "0");
  await flush();
  expect(request.graphs[0].players[0].health).toBe(0);
  expect(request.graphs[0].players[0].is_alive).toBe(false);
});
test("objective and world controls update the actual graph supplied to inference", async () => {
  await mount();
  await click(document.querySelector(".map-objective"));
  await click(
    Array.from(document.querySelectorAll(".check-field"))
      .find((x) => x.textContent === "Destroyed")
      ?.querySelector("input") ?? null,
  );
  await flush();
  expect(request.graphs[0].objectives[0].destroyed).toBe(true);
  expect(request.graphs[0].objectives[0].health).toBe(0);
  await click(document.querySelector(".time-control button"));
  await select("Mid Boss state", "down");
  await numeric("Mid Boss respawn timer", "300");
  await flush();
  expect(request.graphs[0].global_info.mid_boss_up).toBe(false);
  expect(request.graphs[0].global_info.next_midboss).toBe(
    request.graphs[0].elapsed_seconds + 300,
  );
});
test("baseline, presets and ownership changes preserve the real zipline geometry", async () => {
  await mount();
  await click(document.querySelector(".baseline-button"));
  await flush();
  expect(request.graphs).toHaveLength(2);
  const before = request.graphs[0].ziplines[0];
  await click(
    Array.from(document.querySelectorAll(".map-tools>button")).find(
      (x) => x.textContent === "Ziplines",
    ) ?? null,
  );
  await click(
    document.querySelector(
      '.paint-team button[aria-label="Paint Hidden King"]',
    ),
  );
  await click(document.querySelector(".zip-point"));
  await flush();
  expect(request.graphs[0].ziplines[0].team_num).toBe(2);
  expect(request.graphs[0].ziplines[0].connections).toEqual(before.connections);
  expect(request.graphs[0].ziplines[0].x).toBe(before.x);
  await select("Scenario preset", "0");
  await flush();
  expect(request.graphs[0].elapsed_seconds).toBeLessThan(1);
  expect(request.graphs[0].players).toHaveLength(12);
});

test("inventory changes preserve abilities and scenario export is valid JSON", async () => {
  await mount();
  const before = request.graphs[0].players[0].upgrade_ids as number[];
  const equipped = document.querySelector(".inventory button");
  if (equipped) {
    await click(equipped);
    await flush();
    expect((request.graphs[0].players[0].upgrade_ids as number[]).length).toBe(
      before.length - 1,
    );
  }
  const original = URL.createObjectURL,
    originalClick = browser.HTMLAnchorElement.prototype.click;
  let exported: Blob | undefined;
  URL.createObjectURL = (blob) => {
    exported = blob as Blob;
    return "blob:scenario-test";
  };
  browser.HTMLAnchorElement.prototype.click = () => {};
  try {
    await click(document.querySelector(".toolbar-actions button.primary"));
    const payload = JSON.parse(await exported!.text());
    expect(payload.format_version).toBe(1);
    expect(payload.graph.players).toHaveLength(12);
    expect(payload.graph.ziplines).toHaveLength(334);
  } finally {
    URL.createObjectURL = original;
    browser.HTMLAnchorElement.prototype.click = originalClick;
  }
});
