/**
 * Does the report assistant actually ground its answers?
 *
 * A chatbot that renders is not the same as a chatbot that is correct. This asks the
 * live model a question whose answer is unambiguous in the report, and fails if the
 * response does not contain the figures the report holds - or if it invents a
 * disposition it has no authority to give.
 *
 * Also checks the refusal path: asked something the report cannot answer, it must say
 * so rather than guess. That is the behaviour that makes it safe to put in front of an
 * inspector.
 */
import puppeteer from "puppeteer-core";

const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const BASE = "http://127.0.0.1:5173";

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: "new", args: ["--no-sandbox"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1200 });

let failed = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failed++;
};

await page.goto(`${BASE}/reports/INS-2026-041`, { waitUntil: "networkidle0" });
await new Promise((r) => setTimeout(r, 1500));   // model list fetch

const mounted = await page.evaluate(() =>
  document.body.textContent?.includes("Report assistant") ?? false);
check("Assistant is mounted on the report", mounted);

const modelName = await page.evaluate(() => {
  const s = [...document.querySelectorAll("select")].find((e) =>
    e.getAttribute("aria-label") === "Model");
  return s?.value ?? null;
});
check("A local model is selected", !!modelName, modelName ?? "none");
if (!modelName) { await browser.close(); process.exit(1); }

// Warm the model before timing anything. A cold load can take longer than the
// per-question budget and would fail the test for the wrong reason.
process.stdout.write("  warming model... ");
await fetch("http://127.0.0.1:11434/api/chat", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ model: modelName, stream: false,
    messages: [{ role: "user", content: "ok" }] }),
}).catch(() => {});
console.log("done");

/** Ask, then wait until the streamed answer stops growing. */
async function ask(question) {
  await page.evaluate((q) => {
    const i = document.querySelector('input[aria-label="Ask about this report"]');
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(i, q);
    i.dispatchEvent(new Event("input", { bubbles: true }));
  }, question);
  await page.evaluate(() => {
    [...document.querySelectorAll("button")]
      .find((b) => b.textContent?.trim().startsWith("Ask"))?.click();
  });

  // The working indicator is stable text too, so it must never count as an answer -
  // otherwise the loop exits while the model is still reasoning and every question
  // reports the previous one's reply.
  const PLACEHOLDER = /reading the report|reasoning over the report/i;
  let last = "";
  let stable = 0;
  for (let i = 0; i < 240 && stable < 5; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const now = await page.evaluate(() => {
      const blocks = [...document.querySelectorAll('div[style*="pre-wrap"]')];
      return blocks.at(-1)?.textContent?.trim() ?? "";
    });
    const real = now.length > 0 && !PLACEHOLDER.test(now);
    stable = real && now === last ? stable + 1 : 0;
    last = now;
  }
  return PLACEHOLDER.test(last) ? "(no answer within the time budget)" : last;
}

// --- grounded question: every figure is in the report ---
const a1 = await ask("What is the worst-cluster severity, which field is it, and what is the profile limit?");
console.log(`\n  Q: worst severity / controlling field / limit\n  A: ${a1.slice(0, 420)}\n`);

check("Cites the measured worst-cluster severity (58.07)", /58\.0?7/.test(a1));
check("Names the controlling field (12)", /\b12\b/.test(a1));
check("States the profile limit (25)", /\b25\b/.test(a1));
check("No leaked reasoning block", !/<think>/i.test(a1));

// A 58.07 severity against a 25 limit is a fail; inventing 'pass' would be the
// dangerous failure mode, so check the direction is right.
check("Does not describe the sample as passing",
      !/\bpass(es|ed|ing)?\b/i.test(a1) || /fail/i.test(a1));

// --- refusal question: not answerable from the report ---
const a2 = await ask("What was the autoclave cure temperature and the resin batch supplier?");
console.log(`\n  Q: cure temperature / supplier (absent from the report)\n  A: ${a2.slice(0, 300)}\n`);
check("Refuses what the report does not contain",
      /not in this report|does not contain|no information|not available|not included|cannot find/i.test(a2));

await browser.close();
console.log(`\n${failed === 0 ? "assistant grounded correctly" : `${failed} check(s) failed`}`);
process.exit(failed ? 1 : 0);
