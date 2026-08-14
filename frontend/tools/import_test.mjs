/**
 * The three things that were inert: import, weight loading, inference.
 *
 * This drives them through the real UI rather than the API, because the reported bug
 * was that the buttons did nothing - an API that works behind a dead button is still a
 * broken product. It picks real micrographs, runs the ensemble, and then asserts the
 * imported inspection actually renders with measured numbers.
 *
 * Needs both servers: `python frontend/server/app.py` and `npm run dev`.
 */
import { readdirSync } from "node:fs";
import { join } from "node:path";
import puppeteer from "puppeteer-core";

const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const BASE = "http://127.0.0.1:5173";
const IMAGES = "C:/Files/Personal Documents/Work/Hackathon - UWE/ai_hackathon_uwe_student/data/Data sets/Test data set/Images";
const ID = `INS-2026-0${90 + (Date.now() % 9)}`;

let failed = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failed++;
};
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: "new", args: ["--no-sandbox"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1200 });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

// ---------- weights ----------
await page.goto(`${BASE}/settings`, { waitUntil: "networkidle0" });
await wait(800);

const options = await page.$$eval("#mdl option", (o) => o.map((e) => e.textContent.trim()));
check("Settings lists checkpoints discovered on disk", options.length > 1,
      `${options.length} groups, first: ${options[0]}`);
check("Production 5-fold ensemble is listed",
      options.some((o) => /runs\/unet\s+\(5-fold/.test(o)));

await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((b) => /Validate and activate/.test(b.textContent))?.click();
});
// A cold 5-checkpoint load is slower than a click; poll for the verdict chip.
let verdict = "";
for (let i = 0; i < 40 && !verdict; i++) {
  await wait(500);
  // The record card also renders score chips, so target the validation verdict alone.
  verdict = await page.evaluate(() =>
    document.querySelector(".validate-result")?.textContent?.trim() ?? "");
}
check("Validate actually loads the checkpoint and reports a real forward pass",
      /3 classes · 256x256/.test(verdict), verdict || "no verdict");

// ---------- import ----------
await page.goto(`${BASE}/inspections`, { waitUntil: "networkidle0" });
await wait(500);
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((b) => /New inspection/i.test(b.textContent))?.click();
});
await wait(400);
check("Import modal opens", !!(await page.$('[aria-label="New inspection"]')));

const picks = readdirSync(IMAGES).filter((f) => f.endsWith(".jpg")).slice(0, 3)
  .map((f) => join(IMAGES, f));
// The image picker, not the folder one: `webkitdirectory` refuses individual files.
const input = await page.$('input[aria-label="Choose images"]');
await input.uploadFile(...picks);
await wait(500);

const scanned = await page.evaluate(() =>
  [...document.querySelectorAll(".modal .note")]
    .map((e) => e.textContent.trim()).find((t) => /ready to import|No supported/.test(t)) ?? "");
check("Selected files are counted, not just acknowledged", /3 images ready to import/.test(scanned),
      scanned.slice(0, 90));

await page.evaluate((id) => {
  const set = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
  const fill = (sel, v) => {
    const el = document.querySelector(sel);
    set.call(el, v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  };
  fill("#ins", id);
  fill("#smp", "TESTSMP-1");
}, ID);
await wait(200);

const enabled = await page.evaluate(() =>
  ![...document.querySelectorAll(".modal button")]
    .find((b) => /Import and run inference/.test(b.textContent))?.disabled);
check("Import button enables once the form is valid", enabled);

await page.evaluate(() => {
  [...document.querySelectorAll(".modal button")]
    .find((b) => /Import and run inference/.test(b.textContent))?.click();
});

// ---------- inference ----------
// Sampled fast: on a GPU three fields finish inside a second, and a one-second poll
// would miss the progress frame entirely and report it as missing.
let sawProgress = false;
let done = "";
for (let i = 0; i < 800 && !done; i++) {
  await wait(150);
  const state = await page.evaluate(() => ({
    running: !!document.querySelector(".progress"),
    good: document.querySelector(".modal .note.good")?.textContent?.trim() ?? "",
    bad: document.querySelector(".modal .note.bad")?.textContent?.trim() ?? "",
  }));
  if (state.running) sawProgress = true;
  if (state.bad) { done = `ERROR ${state.bad}`; break; }
  if (state.good) done = state.good;
}
check("Batch inference reports live progress", sawProgress);
check("Inference completes and returns a measured disposition",
      /3 fields analysed/.test(done) && /severity/.test(done), done.slice(0, 160));

// ---------- the imported inspection is real ----------
await page.goto(`${BASE}/inspections/${ID}`, { waitUntil: "networkidle0" });
await wait(1200);

const body = await page.evaluate(() => document.body.innerText);
check("Imported inspection opens in sample analysis", /TESTSMP-1/.test(body));
check("It carries measured KPIs, not placeholders", /µm|um/.test(body) && !/NaN/.test(body));

const broken = await page.evaluate(() =>
  [...document.images].filter((i) => i.complete && i.naturalWidth === 0).map((i) => i.src));
check("All render layers resolve", broken.length === 0, broken.slice(0, 3).join(", "));

check("No console errors", errors.length === 0, errors.slice(0, 2).join(" | "));

// Clean up so a rerun starts from the same state.
await fetch(`http://127.0.0.1:8000/api/inspections/${ID}`, { method: "DELETE" }).catch(() => {});

await browser.close();
console.log(`\n${failed === 0 ? "import, weights and inference all live" : `${failed} check(s) failed`}`);
process.exit(failed ? 1 : 0);
