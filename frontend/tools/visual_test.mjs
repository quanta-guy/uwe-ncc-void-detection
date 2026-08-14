/**
 * Automated visual check of every route.
 *
 * Drives the real app in headless Chrome: screenshots each route, fails on console
 * errors, failed network requests, or broken images. Screenshots are written so the
 * result can be judged by eye as well as by assertion - a page can be technically
 * error-free and still be visually broken.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import puppeteer from "puppeteer-core";

const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const BASE = "http://127.0.0.1:5173";
const OUT = "tools/screenshots";

const ROUTES = [
  ["inspections", "/inspections"],
  ["sample-analysis", "/inspections/INS-2026-041"],
  ["field-review", "/inspections/INS-2026-041/fields/12"],
  ["review-queue", "/review"],
  ["reports", "/reports"],
  ["cross-section-report", "/reports/INS-2026-041"],
  ["model-improvement", "/model-improvement"],
  ["settings", "/settings"],
];

mkdirSync(OUT, { recursive: true });
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--force-device-scale-factor=1"],
});

let failures = 0;
const results = [];

for (const [name, route] of ROUTES) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000 });
  const errors = [];
  const badRequests = [];

  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("requestfailed", (r) => badRequests.push(`${r.url()} ${r.failure()?.errorText}`));
  page.on("response", (r) => { if (r.status() >= 400) badRequests.push(`${r.status()} ${r.url()}`); });

  await page.goto(BASE + route, { waitUntil: "networkidle0", timeout: 30000 });
  await new Promise((r) => setTimeout(r, 700));   // let charts finish animating in

  // Broken images are invisible to console errors but obvious to a viewer.
  const broken = await page.evaluate(() =>
    [...document.images].filter((i) => !i.complete || i.naturalWidth === 0).map((i) => i.src));

  const heading = await page.evaluate(() =>
    document.querySelector("h1")?.textContent?.trim() ?? "(no h1)");
  const activeNav = await page.evaluate(() =>
    document.querySelector("nav a.active")?.textContent?.trim() ?? "(none)");
  const imgCount = await page.evaluate(() => document.images.length);

  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });

  const ok = errors.length === 0 && broken.length === 0 && badRequests.length === 0;
  if (!ok) failures++;
  results.push({ name, route, heading, activeNav, imgCount, errors, broken, badRequests, ok });

  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(22)} h1="${heading}" nav="${activeNav}" imgs=${imgCount}`);
  for (const e of errors) console.log(`        console: ${e}`);
  for (const b of broken) console.log(`        broken image: ${b}`);
  for (const b of badRequests) console.log(`        request: ${b}`);
  await page.close();
}

await browser.close();
writeFileSync(`${OUT}/results.json`, JSON.stringify(results, null, 2));
console.log(`\n${ROUTES.length - failures}/${ROUTES.length} routes clean -> ${OUT}/`);
process.exit(failures ? 1 : 0);
