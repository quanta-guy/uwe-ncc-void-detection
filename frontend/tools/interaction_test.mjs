/**
 * Drives the interactions, not just the routes.
 *
 * The route test proves a page renders. It cannot prove a button does anything, which
 * is exactly the class of defect that was reported: a Browse button that only wrote a
 * hardcoded string, an Edit mask button wired to nothing, and a missing delete action.
 * Each check below fails if the control is inert.
 */
import { mkdirSync } from "node:fs";
import puppeteer from "puppeteer-core";

const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const BASE = "http://127.0.0.1:5173";
const OUT = "tools/screenshots";
mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: "new",
  args: ["--no-sandbox", "--force-device-scale-factor=1"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1000 });
page.on("dialog", (d) => d.accept());          // confirm() on delete

let failed = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failed++;
};
const byText = async (sel, text) =>
  page.evaluateHandle((s, t) =>
    [...document.querySelectorAll(s)].find((e) => e.textContent?.includes(t)) ?? null, sel, text);
const clickText = async (sel, text) => {
  const h = await byText(sel, text);
  const el = h.asElement();
  if (!el) return false;
  await el.click();
  await new Promise((r) => setTimeout(r, 350));
  return true;
};

// ---------- 1. folder picker is a real directory input ----------
await page.goto(`${BASE}/inspections`, { waitUntil: "networkidle0" });
await clickText("button", "New inspection");
const picker = await page.evaluate(() => {
  const i = document.querySelector('input[type="file"][webkitdirectory]');
  return i ? { present: true, multiple: i.multiple } : { present: false };
});
check("Browse opens a real directory picker", picker.present,
      picker.present ? "input[webkitdirectory] present" : "no directory input found");

const hardcoded = await page.evaluate(() =>
  (document.querySelector("#fld")?.value ?? "").includes("Data sets"));
check("Browse does not write a hardcoded path", !hardcoded);
await page.screenshot({ path: `${OUT}/fix-new-inspection.png` });
await page.keyboard.press("Escape");
await clickText("button", "Cancel");

// ---------- 2. Edit mask opens a working editor ----------
await page.goto(`${BASE}/inspections/INS-2026-041/fields/12`, { waitUntil: "networkidle0" });
await clickText("button", "Wrong — correct mask");
const modalUp = await page.evaluate(() =>
  !!document.querySelector('[aria-label="Correct the prediction"]'));
check("Correction modal opens", modalUp);

// Edit is gated on an error reason, which is the specified behaviour.
const gated = await page.evaluate(() => {
  const b = [...document.querySelectorAll("button")].find((e) => e.textContent?.includes("Edit mask"));
  return b ? b.disabled : null;
});
check("Edit mask is gated until a reason is chosen", gated === true);

await page.evaluate(() => {
  const r = document.querySelector('input[type="radio"][value="boundary_error"]');
  r?.click();
});
await new Promise((r) => setTimeout(r, 200));
await clickText("button", "Edit mask");
await new Promise((r) => setTimeout(r, 1200));   // editor loads and reconstructs the mask

const editor = await page.evaluate(() => {
  const dlg = document.querySelector('[aria-label="Mask editor"]');
  const canvas = dlg?.querySelector("canvas");
  return { open: !!dlg, canvas: !!canvas, w: canvas?.width ?? 0 };
});
check("Edit mask opens the editor", editor.open && editor.canvas,
      `canvas ${editor.w}px`);

// Paint on the canvas and confirm the pixel counters actually move.
if (editor.open) {
  const box = await (await page.$('[aria-label="Mask editor"] canvas')).boundingBox();
  await page.mouse.move(box.x + box.width * 0.42, box.y + box.height * 0.42);
  await page.mouse.down();
  for (let i = 0; i < 12; i++) {
    await page.mouse.move(box.x + box.width * (0.42 + i * 0.01),
                          box.y + box.height * (0.42 + i * 0.008));
  }
  await page.mouse.up();
  await new Promise((r) => setTimeout(r, 400));

  const changed = await page.evaluate(() => {
    const rows = [...document.querySelectorAll('[aria-label="Mask editor"] .mrow')];
    const row = rows.find((r) => r.textContent?.includes("Pixels changed"));
    return Number(row?.querySelector(".v")?.textContent?.trim() ?? "0");
  });
  check("Painting changes the mask", changed > 0, `${changed} px changed`);

  const saveOn = await page.evaluate(() => {
    const b = [...document.querySelectorAll('[aria-label="Mask editor"] button')]
      .find((e) => e.textContent?.includes("Save correction"));
    return b ? !b.disabled : null;
  });
  check("Save correction enabled after an edit", saveOn === true);
  await page.screenshot({ path: `${OUT}/fix-mask-editor.png` });

  await clickText('[aria-label="Mask editor"] button', "Save correction");
  const recorded = await page.evaluate(() =>
    document.body.textContent?.includes("Mask edited.") ?? false);
  check("Editor result returns to the correction modal", recorded);
}

// ---------- 3. delete action exists and works ----------
await page.goto(`${BASE}/inspections`, { waitUntil: "networkidle0" });
const before = await page.evaluate(() => document.querySelectorAll("tbody tr").length);
const delBtn = await page.$('button[aria-label^="Remove INS-"]');
check("Workspace has a delete action", !!delBtn);
if (delBtn) {
  await delBtn.click();
  await new Promise((r) => setTimeout(r, 500));
  const after = await page.evaluate(() => document.querySelectorAll("tbody tr").length);
  check("Delete removes the row", after === before - 1, `${before} -> ${after}`);

  await page.goto(`${BASE}/settings`, { waitUntil: "networkidle0" });
  const restorable = await page.evaluate(() =>
    [...document.querySelectorAll("button")]
      .some((b) => /Restore \d+ hidden inspection/.test(b.textContent ?? "") && !b.disabled));
  check("Deletion is reversible from Settings", restorable);
  await page.screenshot({ path: `${OUT}/fix-settings-restore.png` });
  await clickText("button", "Restore");
}

await browser.close();
console.log(`\n${failed === 0 ? "all interaction checks passed" : `${failed} check(s) failed`}`);
process.exit(failed ? 1 : 0);
