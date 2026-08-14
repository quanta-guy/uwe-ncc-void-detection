# Composite Inspection Prototype

Frontend product, UX and implementation specification for the CFRP microscopy inspection prototype.

This document is the build contract for the prototype. It combines the agreed product workflow, screen designs, measurable KPIs, model-review workflow, sample data and implementation plan. It intentionally uses **filesystem-backed JSON and image artifacts instead of a database**.

## 1. Product outcome

The product is not a binary defect classifier. It is a local inspection workspace that turns microscopy images into:

- Quantified material and void measurements.
- A prioritised human-review queue.
- Corrected and traceable segmentation masks.
- A sample-level quality disposition with its controlling evidence.
- A reviewed dataset that can be exported for calibration or model fine-tuning.

The prototype should demonstrate the complete path from imported image folder to an evidence-backed report:

```text
Import images
    -> mask every image automatically
    -> calculate preliminary measurements
    -> prioritise fields for human review
    -> accept or correct each mask
    -> recalculate from the accepted masks
    -> approve and export the cross-section report
    -> retain reviewed feedback for later model improvement
```

### Business value demonstrated

1. Reviewers inspect the most consequential fields first instead of searching every image manually.
2. Decisions are supported by measurements and visual evidence rather than only `defect / no defect`.
3. The application creates a reproducible inspection record: original image, model version, prediction, correction, reviewer decision and final measurements.
4. Corrections become reusable labelled data without changing the production model during an inspection.
5. Sample and batch trends can later expose recurring material, supplier or process problems.

## 2. Prototype boundary

### Included

- Local Windows-first web application.
- Folder import and folder/table workspace views.
- Automatic batch inference immediately after import.
- Sample analysis and field risk ranking.
- Side-by-side, swipe and overlay image comparison.
- Hue, brightness and contrast controls for human inspection only.
- Accept-mask and wrong/correct-mask review actions.
- Mask editing, single-mask upload and bulk-mask upload.
- Field and sample KPIs.
- Reports dashboard and detailed cross-section quality report.
- Model path, calibration, measurement-profile and output-folder settings.
- Filesystem-backed review history and correction dataset.
- Calibration and fine-tuning entry points that produce candidates, never automatic production changes.

### Deliberately excluded from this prototype

- Database.
- User accounts, roles or single sign-on.
- Cloud upload or cloud storage.
- Redis, Celery or another queue service.
- Microservices.
- Concurrent multi-user editing.
- Automatic online learning.
- Automatic deployment of a newly tuned model.
- Mobile layouts.
- Certification to a material standard.

Add a database only when more than one workstation or reviewer needs to edit the same inspections concurrently. Until then, JSON files are easier to inspect, demo, copy and recover.

## 3. Terminology

Use this hierarchy everywhere:

```text
Inspection -> Sample -> Cross-section -> Field -> Void
```

| Term | Meaning |
|---|---|
| Inspection | The complete work item and its lifecycle. |
| Sample | The user-facing replacement for “specimen”. |
| Cross-section | The polished microscopy surface represented by a set of fields. |
| Field | One microscopy image or tile. |
| Void | A connected void region in an accepted segmentation mask. |
| Measurement profile | The named set of business limits used to calculate a disposition. It is not the sample itself. |
| Prediction | The immutable mask produced by a particular model version. |
| Accepted mask | Either the accepted prediction or the latest saved human correction. |

Do not use “specimen” in the interface. Do not use “profile” as a synonym for a folder or sample.

## 4. Product truths and guardrails

### Material type is an input, not a model conclusion

Do not infer C/PEEK versus C/LM-PEAK from fibre radius in this prototype. Apparent radius in pixels is strongly affected by microscope scale. The supplied test data has the same 7 px average fibre radius and 0.57 µm/px calibration for both materials. Across the training data, `fibre_radius_px * µm_per_pixel` is approximately a constant physical fibre radius.

Material must therefore be selected during import or read from trusted metadata. A future material-suggestion feature would require an independently labelled validation study and should still be shown as a suggestion, not inspection truth. See the [dataset notes](<../data/READ ME.txt>).

### Measurements require calibration

Micron and square-millimetre outputs require a trusted `µm/pixel` value. If calibration is missing:

- Show pixel-based measurements only.
- Mark the sample `Uncalibrated`.
- Do not calculate a physical pass/fail disposition.
- Keep the calibration control visible; this is a physical measurement system.

### Model output is preliminary

Before human review, every KPI and disposition must be labelled **Model-derived / human review pending**. After all required fields are reviewed, measurements must be recalculated from the accepted masks.

### Pass/fail needs an approved profile

The current 25 µm cluster-severity limit comes from the challenge logic; it is not a universal composite-material standard. If no approved measurement profile is selected, show `Measured — no disposition` instead of inventing a pass/fail result.

### Feedback does not update the live model

Accepting or correcting a mask records evidence. It does not update weights in memory. Model improvement is a separate, versioned and validated workflow.

## 5. Recommended prototype stack

### Frontend

- React with TypeScript.
- Vite.
- React Router for the five top-level routes.
- Plain CSS modules and CSS variables; no component framework.
- Lucide React for consistent icons.
- Recharts for the disposition donut, field-severity chart and trend charts.
- React Konva/Konva for pan, zoom, overlays, brush, eraser and mask editing.
- Native `fetch` and React state for API data; add a query library only if state synchronisation becomes a real problem.

### Local application layer

- Python and FastAPI, reusing the repository's PyTorch inference and measurement code.
- Uvicorn bound to `127.0.0.1` only.
- One local inference process, launched by the application.
- pywebview/WebView2 for the final Windows desktop wrapper after the browser version works.
- PyInstaller one-directory packaging only after the workflow is stable.

### Existing ML and measurement libraries

- PyTorch.
- NumPy.
- Pillow.
- SciPy.
- scikit-image.
- OpenCV-headless where contour or Feret geometry requires it.

Do not add PostgreSQL, SQLite, Electron, Tauri, Redis, Celery or MLflow to the prototype.

## 6. Application navigation

Use a dark fixed top navigation bar with these destinations:

1. **Inspections** — folder/table workspace and sample analysis.
2. **Review queue** — field-level validation and mask correction.
3. **Reports** — aggregate dashboard and detailed cross-section reports.
4. **Model improvement** — reviewed dataset, calibration, fine-tuning and candidate comparison.
5. **Settings** — active model path, output path and defaults.

The current mock-up label `Learning` should become `Model improvement`. `Administration` should become `Settings` for the prototype.

### Routes

```text
/inspections
/inspections/:inspectionId
/inspections/:inspectionId/fields/:fieldId
/review
/reports
/reports/:inspectionId
/model-improvement
/settings
```

## 7. Visual system

The interface should feel like an industrial measurement tool: dense, calm, evidence-led and usable for long review sessions.

### Design tokens

```css
:root {
  --nav: #061622;
  --nav-active: #00c7c7;
  --teal-700: #006f75;
  --teal-600: #00858b;
  --teal-100: #dff5f3;
  --surface: #ffffff;
  --canvas: #f5f7f8;
  --border: #d6dee3;
  --text: #101820;
  --muted: #5f6f7d;
  --success: #258a38;
  --warning: #d99000;
  --danger: #d51f26;
  --info: #1473e6;
  --radius: 6px;
  --shadow: 0 2px 10px rgb(10 35 50 / 8%);
}
```

### Typography and spacing

- Use the offline system stack: `"Segoe UI", Arial, sans-serif`.
- Page title: 30–36 px, semibold.
- Section title: 18–22 px, semibold.
- Body: 14–16 px.
- KPI number: 28–44 px depending on hierarchy.
- Base spacing: 8 px; common gaps: 8, 16, 24 and 32 px.
- Minimum interactive target: 40 × 40 px.

### Layout

- Desktop target: 1440–1920 px wide.
- Minimum supported viewport: 1024 px.
- Use a 12-column content grid.
- Field review reserves approximately 20% for the queue, 60% for images and 20% for measurements at wide widths.
- Below 1280 px, collapse the queue and move measurements below the image comparison.
- Do not design a mobile version for the prototype.

### Accessibility

- Never communicate status by colour alone; pair colour with text and an icon.
- Keep visible keyboard focus.
- Provide image and control labels.
- Use explicit button text such as `Accept mask` and `Wrong — correct mask`; thumbs icons may accompany the text.
- Ensure warning and danger text meets contrast requirements on tinted backgrounds.

## 8. Screen specifications

### 8.1 Inspection workspace

Purpose: find work, import a folder and understand operational state.

![Inspection workspace reference](../../product-design/v2.1/01-inspection-workspace.png)

Required UI:

- `New inspection` button.
- Search.
- Filters: All, Processing, Ready for review and Completed.
- View switch: `Folder` and `Table`.
- Folder tree showing inspection -> sample -> cross-section -> fields.
- Inspection table with columns: Inspection, Sample, Material, Fields, Processing, Review, Preliminary result and Model.
- Selected-inspection drawer with sample metadata, progress timeline and only the controlling preliminary result.

Operational KPIs shown here only:

- Active inspections.
- Images processing.
- Ready for review.
- Completed today.

Do not crowd this screen with the complete material KPI set.

#### New inspection flow

The import modal collects:

- Inspection ID, generated by default and editable.
- Sample ID.
- Image folder.
- Material: C/PEEK, C/LM-PEAK or another explicitly entered value.
- Calibration in µm/pixel.
- Measurement profile.
- Model version, defaulting to the active validated model.

When the user confirms:

1. Validate supported files and duplicate stems.
2. Copy source images into the inspection's immutable `originals` folder.
3. Write the inspection manifest.
4. Start batch inference automatically.
5. Show live `processed / total` progress by polling `status.json`.
6. Enable review only after predictions and preliminary measurements exist.

### 8.2 Sample analysis

Purpose: understand the cross-section result and choose where review should start.

![Sample analysis reference](../../product-design/v2.1/02-sample-analysis-kpis-v2.png)

Required UI:

- Sample identity, material, number of fields, calibration and model version.
- Dominant preliminary-quality strip.
- Field risk map made from real thumbnails.
- Review-priority table.
- `Start priority review` and `Sequential review` actions.
- Secondary composition and void-distribution cards.
- Action menu containing `Bulk mask upload`.

Primary sample KPIs:

- Void areal fraction.
- Worst-cluster severity.
- Fields over profile limit.
- Largest void Feret diameter.

Secondary sample KPIs:

- Fibre areal fraction.
- Matrix areal fraction.
- Void count.
- Void density.
- Void Feret p50 and p95.
- Maximum void Feret diameter.
- Percentage of sampled area/fields exceeding the profile.

The risk map uses `Routine`, `Review` and `Critical`, not pass/fail colours without explanation. Clicking a field opens it directly.

### 8.3 Field review

Purpose: validate one prediction quickly and record a trustworthy accepted mask.

![Field review reference](../../product-design/v2.1/03-field-review-kpis-v2.png)

Required UI:

- Priority or sequential queue in the left panel.
- Original and prediction images with synchronised pan and zoom.
- View modes: Side by side, Swipe and Overlay.
- Mask-opacity control.
- Original-image hue, brightness and contrast controls.
- Class toggles: Matrix, Fibre, Void and Controlling cluster.
- `Accept mask`, `Wrong — correct mask` and `Mark unsuitable` above the images.
- Previous and `Save & next` navigation.
- Optional review note.
- Active-field measurement panel.

Image adjustments are display-only. They must never modify the source file, model input, mask or measurements.

Field KPIs:

- Field void areal fraction.
- Field void count.
- Largest void Feret diameter.
- Cluster severity.
- Distance to the active profile limit.
- Model disagreement or confidence status when available.
- Review status.
- Calibration.

Do not show sample-wide distributions or training metrics on this screen.

#### Keyboard workflow

```text
A          accept mask
W          open correction modal
U          mark unsuitable
Left/Right previous/next field
Space      toggle overlay
+ / -      zoom
0          reset view and display adjustments
```

Shortcuts must be disabled while typing in a note or modal.

### 8.4 Wrong/correction modal and mask editor

The edit and single-mask upload controls appear only after `Wrong — correct mask` is selected.

![Correction modal reference](../../product-design/v1/02-wrong-correction-modal.png)

The modal asks for an error reason:

- Missed void.
- False void.
- Boundary error.
- Fibre/matrix classification error.
- Input-quality problem.
- Other, with a note.

Actions:

- `Edit mask` opens the mask editor.
- `Upload corrected mask` selects one file.
- `Cancel` returns without changing the review decision.

Mask editor requirements:

- Start from a copy of the original prediction.
- Brush and eraser.
- Class selector for matrix, fibre and void.
- Brush-size control.
- Pan and zoom.
- Undo and redo.
- Overlay-opacity control.
- Reset to original prediction.
- Save correction.

On save:

1. Validate image dimensions and class values `0, 1, 2`.
2. Write a new versioned correction PNG; never overwrite the prediction.
3. Recalculate the field measurements.
4. Record the review event.
5. Update sample aggregates.
6. Move to the next field.

### 8.5 Bulk mask upload

Bulk upload belongs in Sample analysis or the review-queue action menu. It is separate from the single-field correction modal.

Expected input:

- A selected folder or ZIP containing PNG masks.
- Mask filenames matching source-image stems.

Preview before committing:

- Matched masks.
- Missing masks.
- Unknown files.
- Duplicate stems.
- Wrong dimensions.
- Invalid class values.
- Existing corrections that would receive a new version.

Never silently overwrite an existing prediction or correction. Commit valid masks only after the user confirms the mapping summary.

### 8.6 Reports dashboard

Purpose: provide operational value across completed inspections.

Use the V1 dashboard as the aggregate-report reference:

![Reports dashboard reference](../../product-design/v1/03-final-reports.png)

Required KPIs:

- Inspections/samples completed.
- Fields reviewed.
- Correction rate.
- Pass/fail/review percentage as a donut chart.
- Accepted-first-time percentage.
- Corrected-mask count.
- Void-content trend by completed sample.
- Completed-sample table with material, fields, disposition and corrections.

Clicking a sample opens its detailed cross-section report.

### 8.7 Cross-section quality report

Purpose: show the complete evidence package for one sample.

![Cross-section quality report reference](../../product-design/v2.1/04-cross-section-quality-report-v2.png)

This is the only inspection screen with the complete final KPI set.

Sections:

1. Final or preliminary disposition, reason and approval status.
2. Decision KPI strip.
3. Spatial field evidence.
4. Cluster severity by field with the profile limit.
5. Controlling original and accepted-mask overlay.
6. Composition and morphology.
7. Traceability and assurance.
8. Review decisions and exceptions.
9. Approval timeline.

Exports:

- Printable HTML report.
- CSV measurement table.
- JSON evidence manifest.
- PDF may be produced by the browser print dialog for the prototype.

The exported evidence manifest records:

- Inspection and sample IDs.
- Material selection.
- Calibration.
- Measurement profile and limits.
- Model filename/version and SHA-256 checksum.
- Original-image checksums.
- Fields reviewed and corrected.
- Accepted-mask version for each field.
- Final measurements.
- Reviewer name entered for the demo and approval timestamp.

### 8.8 Model improvement

Purpose: use reviewed corrections safely without turning the inspection application into an uncontrolled online-learning system.

![Model improvement reference](../../product-design/v2.1/05-model-improvement.png)

Prototype actions:

- List accepted predictions and corrected masks from inspection folders.
- Filter by material, error reason and date.
- Export a frozen reviewed-dataset snapshot.
- Run post-processing calibration against a selected reviewed dataset.
- Optionally launch the existing fine-tuning script as a separate process.
- Display candidate-versus-production validation results.
- Allow manual promotion only after validation succeeds.

Recommended order:

1. Calibrate threshold and minimum-object-size settings.
2. Inspect whether reviewed errors remain systematic.
3. Fine-tune weights from the current approved checkpoint only if calibration cannot fix them.
4. Mix reviewed corrections with representative replay data to reduce catastrophic forgetting.
5. Validate on a locked holdout grouped by source cross-section.
6. Report C/PEEK and C/LM-PEAK independently.
7. Promote by writing a new active-model setting; never overwrite the old model.

Model KPIs shown only here:

- Void Dice on void-containing fields.
- Critical-failure recall.
- False-accept rate.
- False-reject rate.
- Manual-correction rate.
- Review-queue rate.
- Per-material performance.

The current production model should remain active. The `solution3` canonical-spacing candidate should not replace it because it missed two visible C/LM-PEAK voids in the hidden-style test set. Retain its improvements to grouped validation, original-only validation, live augmentation, fold balancing and nested evaluation, but correct the evaluator's target-spacing behaviour and validate any next candidate before use. See [solution3 notes](../solution3/README.md).

### 8.9 Settings

Use the model-settings design as the reference, simplified for filesystem persistence:

![Settings reference](../../product-design/v1/04-model-settings.png)

Required settings:

- Active model/checkpoint path.
- `Validate model` action.
- Model ID/version shown from its manifest or filename.
- Default calibration.
- Default measurement profile.
- Prototype data/output folder.
- Automatically run batch inference after import.
- Keep original predictions after corrections; this is always on and cannot be disabled.
- Reviewed-feedback folder and counts.
- Export corrections.

Selecting a file does not immediately activate it. Validation must first confirm that the checkpoint loads and returns a three-class mask with the expected dimensions.

## 9. KPI definitions and placement

### Material and defect measurements

| KPI | Calculation | UI placement |
|---|---|---|
| Void areal fraction (%) | Void pixels / valid sampled pixels × 100. | Sample analysis, field review, report. |
| Fibre areal fraction (%) | Fibre pixels / valid sampled pixels × 100. | Sample analysis secondary card, report. |
| Matrix areal fraction (%) | Matrix pixels / valid sampled pixels × 100. | Sample analysis secondary card, report. |
| Void count | Connected void regions after the approved cleanup rule. | Field review, report. |
| Void density (/mm²) | Void count / calibrated sampled area. | Sample analysis, report. |
| Feret p50/p95/max (µm) | Percentiles and maximum of calibrated void Feret diameters. | Sample analysis distribution, report. |
| Worst-cluster severity | Maximum challenge-defined severity across grouped voids. | Workspace controlling result, sample analysis, field review, report. |
| Fields over limit | Fields where the controlling metric exceeds the active profile. | Sample analysis, report. |
| Percentage sampled area/fields over limit | Exceeding fields / valid fields × 100 for this prototype. | Report. |
| Worst-field annotated image | Original field paired with the accepted mask and controlling-cluster outline. | Detailed report and export. |
| Measurement confidence/review status | Calibration, model disagreement, review and input-quality state. | Field review and report traceability. |

The current challenge severity calculation is:

```text
cluster severity = sum(void diameters in cluster)
                 + 0.5 * sqrt(total void area in cluster)
```

Void regions separated by less than 40 µm are grouped using single-linkage clustering. Keep this logic in Python as the single measurement source. Label it `Challenge cluster severity` or the selected profile's approved business name; do not present it as a universal standard metric.

### Model and workflow KPIs

| KPI | Placement |
|---|---|
| Images processed / total | Workspace and sample status. |
| Fields reviewed / total | Workspace, sample status and report. |
| Correction rate | Reports dashboard and Model improvement. |
| Review-queue rate | Model improvement. |
| Void Dice | Model improvement only. |
| Critical-failure recall | Model improvement only. |
| False-accept / false-reject rates | Model improvement only. |

### Interpretation restrictions

- Two-dimensional fibre area fraction may approximate volume fraction only under valid stereological assumptions. Do not label it certified fibre volume fraction in the prototype.
- Model confidence is a review-prioritisation signal, not a material-quality KPI.
- Image-quality warnings are not material failures.
- A sample result is only as representative as its field sampling plan.

## 10. Sample content for the prototype

### Primary demonstration sample

Use the 16-field C/PEEK cross-section `2-6-1_mid` from the test dataset. It has a known calibration of 0.57 µm/px and produces a useful mix of routine and critical fields.

Provisional model-derived sample values currently used by the V2.1 screens:

```text
Void areal fraction:       0.555%
Fibre areal fraction:     55.106%
Matrix areal fraction:    44.339%
Void count:                8
Void density:             23.5/mm²
Feret p50:                20.75 µm
Feret p95:                30.81 µm
Largest void Feret:       31.00 µm
Worst-cluster severity:   59.34 µm
Fields over 25 µm limit:   4 / 16
Critical fields:          03, 11, 12 and 13
Controlling field:        12
```

These are design/demo values derived from current prediction masks, not approved inspection results. Human review starts at `0 / 16`.

#### Field 12: controlling C/PEEK example

Use this image for field review, correction and controlling-evidence screens:

![C/PEEK Field 12 original](<../data/Data sets/Test data set/Images/2-6-1_mid_768_1536.jpg>)

- Original: [2-6-1_mid_768_1536.jpg](<../data/Data sets/Test data set/Images/2-6-1_mid_768_1536.jpg>)
- Prediction: [2-6-1_mid_768_1536.png](../predicted_masks/2-6-1_mid_768_1536.png)
- Intended UI field number: 12.
- Example field values: void fraction 3.384%, void count 2, largest Feret 31.00 µm and cluster severity 59.34 µm.

#### Field 03: obvious void and bulk-mask example

![C/PEEK Field 03 original](<../data/Data sets/Test data set/Images/2-6-1_mid_512_0.jpg>)

- Original: [2-6-1_mid_512_0.jpg](<../data/Data sets/Test data set/Images/2-6-1_mid_512_0.jpg>)
- Prediction: [2-6-1_mid_512_0.png](../predicted_masks/2-6-1_mid_512_0.png)
- Intended UI field number: 03.

### C/LM-PEAK correction example

Use `17-5-2_zoom_mid` to demonstrate a second material and a case that needs human attention. The existing production ensemble reported high disagreement on the following field; later candidates also missed visible C/LM-PEAK voids, so it is appropriate correction-workflow evidence.

![C/LM-PEAK human-review example](<../data/Data sets/Test data set/Images/17-5-2_zoom_mid_512_768.jpg>)

- Original: [17-5-2_zoom_mid_512_768.jpg](<../data/Data sets/Test data set/Images/17-5-2_zoom_mid_512_768.jpg>)
- Prediction: [17-5-2_zoom_mid_512_768.png](../predicted_masks/17-5-2_zoom_mid_512_768.png)
- Use: wrong/correct-mask demonstration and reviewed-dataset export.

### Mask rendering

Stored masks use class values:

```text
0 = matrix
1 = fibre
2 = void
```

They appear nearly black in an ordinary image viewer because values 0–2 are intensity values, not display colours. The frontend must apply a palette at render time:

```text
matrix       transparent or neutral grey
fibre        teal at adjustable opacity
void         red at adjustable opacity
controlling  yellow outline, calculated from the measurement result
```

Use real images and real masks in the prototype. The microscopy shown inside the generated mock-ups is illustrative and must not be treated as source evidence.

## 11. Filesystem persistence instead of a database

### Prototype data layout

```text
prototype-data/
  settings.json
  profiles/
    challenge-severity-v1.json
  models/
    production-v1/
      model.pt
      manifest.json
      validation.json
  inspections/
    INS-2026-041/
      inspection.json
      status.json
      originals/
        2-6-1_mid_*.jpg
      thumbnails/
        2-6-1_mid_*.jpg
      predictions/
        production-v1/
          masks/
            2-6-1_mid_*.png
          probability/
            2-6-1_mid_*.npz
      corrections/
        2-6-1_mid_768_1536/
          v001.png
          v001.json
      measurements/
        preliminary.json
        accepted.json
      reviews.jsonl
      report/
        report.html
        measurements.csv
        evidence.json
```

### Write rules

- Copy imports into `originals`; never modify them.
- Store SHA-256 hashes in `inspection.json`.
- Keep predictions under a model-version directory.
- Corrections are new versions, never overwrites.
- Append review events to `reviews.jsonl`.
- Write JSON atomically: write `file.tmp`, flush, then replace the target.
- Rebuild aggregate dashboards by scanning manifests when the application starts or the Reports page opens.
- One application instance owns a prototype-data directory. A simple lock file prevents two instances from writing simultaneously.

### Minimal `settings.json`

```json
{
  "activeModelPath": "models/production-v1/model.pt",
  "dataRoot": "prototype-data",
  "defaultCalibrationUmPerPixel": 0.57,
  "defaultProfile": "challenge-severity-v1",
  "autoRunInference": true
}
```

### Minimal measurement profile

```json
{
  "id": "challenge-severity-v1",
  "name": "Challenge cluster severity v1",
  "clusterSeverityLimitUm": 25.0,
  "mergeDistanceUm": 40.0,
  "areaFactor": 0.5,
  "approvedForProductionUse": false,
  "note": "Prototype challenge rule; replace with an approved business profile."
}
```

### Minimal inspection manifest

```json
{
  "schemaVersion": 1,
  "inspectionId": "INS-2026-041",
  "sampleId": "CPEEK-041",
  "material": "C/PEEK",
  "calibrationUmPerPixel": 0.57,
  "measurementProfile": "challenge-severity-v1",
  "model": {
    "id": "production-v1",
    "path": "models/production-v1/model.pt",
    "sha256": "demo-value"
  },
  "state": "ready_for_review",
  "fields": [
    {
      "id": "12",
      "stem": "2-6-1_mid_768_1536",
      "original": "originals/2-6-1_mid_768_1536.jpg",
      "prediction": "predictions/production-v1/masks/2-6-1_mid_768_1536.png",
      "acceptedMask": null,
      "reviewStatus": "pending"
    }
  ]
}
```

### Review event

```json
{"at":"2026-08-14T10:42:00+01:00","fieldId":"12","decision":"corrected","errorReason":"boundary_error","acceptedMask":"corrections/2-6-1_mid_768_1536/v001.png","reviewer":"Demo reviewer"}
```

The frontend should not edit these files directly. FastAPI owns validation and writes; the frontend receives JSON over the local API.

## 12. Minimal API contract

```text
GET    /api/settings
PUT    /api/settings
POST   /api/models/validate

GET    /api/inspections
POST   /api/inspections/import
GET    /api/inspections/:id
GET    /api/inspections/:id/status
GET    /api/inspections/:id/fields
GET    /api/inspections/:id/fields/:fieldId

GET    /api/assets?inspection=:id&path=:relativePath
POST   /api/inspections/:id/fields/:fieldId/accept
POST   /api/inspections/:id/fields/:fieldId/unsuitable
POST   /api/inspections/:id/fields/:fieldId/corrections
POST   /api/inspections/:id/masks/bulk-validate
POST   /api/inspections/:id/masks/bulk-commit

GET    /api/reports
GET    /api/reports/:inspectionId
POST   /api/reports/:inspectionId/approve
GET    /api/reports/:inspectionId/export/:format

GET    /api/model-improvement/dataset
POST   /api/model-improvement/export
POST   /api/model-improvement/calibrate
POST   /api/model-improvement/fine-tune
GET    /api/model-improvement/runs/:runId
POST   /api/model-improvement/runs/:runId/promote
```

Use ordinary one-second polling for inference/tuning progress. WebSockets are unnecessary for the prototype.

## 13. Frontend structure

Keep the implementation small and feature-oriented:

```text
frontend/
  README.md
  package.json
  vite.config.ts
  src/
    main.tsx
    app.tsx
    api.ts
    types.ts
    styles.css
    components/
      AppShell.tsx
      StatusChip.tsx
      KpiStrip.tsx
      ImageViewer.tsx
      MaskEditor.tsx
      FieldGrid.tsx
      CorrectionModal.tsx
    pages/
      InspectionsPage.tsx
      SampleAnalysisPage.tsx
      FieldReviewPage.tsx
      ReportsPage.tsx
      CrossSectionReportPage.tsx
      ModelImprovementPage.tsx
      SettingsPage.tsx
    fixtures/
      inspections.json
      report.json
```

Do not create a global-state framework, design-system package or generic repository layer for the prototype. Shared components should exist only when at least two screens use them.

## 14. State machines

### Inspection

```text
importing -> processing -> ready_for_review -> in_review -> ready_for_approval -> complete
                  \-> failed
```

### Field review

```text
pending -> accepted
pending -> corrected
pending -> unsuitable
```

### Model candidate

```text
candidate -> validating -> validated -> approved -> active -> retired
                       \-> rejected
```

The UI must display recovery actions for `failed` imports or inference runs. It must never quietly treat a failed field as a no-void field.

## 15. Input validation

### Imported images

- Allow `.png`, `.jpg`, `.jpeg`, `.tif` and `.tiff`.
- Reject empty folders.
- Reject duplicate stems.
- Record dimensions and checksum.
- Flag unreadable files individually.
- Never infer calibration from the filename.

### Masks

- Dimensions must match the source field.
- Must be a single-channel integer mask after normalisation.
- Allowed values are exactly 0, 1 and 2.
- Filename stem must map to one field.
- A new correction version is created if one already exists.

### Model checkpoint

- Path must exist and be readable.
- Load only trusted internal checkpoints.
- Validate expected architecture/classes and a dry-run tensor.
- Store the checksum and resolved version in the inspection manifest.

## 16. Implementation plan

### Phase 0 — contracts and fixtures

- Create TypeScript types for settings, inspection, field, measurement and report.
- Create fixture JSON using `INS-2026-041` and the provisional values in this README.
- Render real sample images and palette-rendered masks.
- Agree the JSON/API contract before connecting inference.

Done when all screens can be navigated with consistent fixture data and no broken image paths.

### Phase 1 — application shell and inspection workspace

- Build navigation and visual tokens.
- Build folder/table views, filters and details drawer.
- Build the new-inspection modal.
- Connect settings and folder import.
- Write `inspection.json` and `status.json` through FastAPI.

Done when a real folder can be imported and appears as `Processing`.

### Phase 2 — batch inference and sample analysis

- Wrap existing prediction code in one local inference process.
- Produce masks, thumbnails, probability artifacts and preliminary measurements.
- Poll progress.
- Build the sample KPI strip, field risk map and priority table.

Done when every imported image is masked before the review screen becomes available.

### Phase 3 — field review and corrections

- Build synchronised image viewers and class palette.
- Add accept, unsuitable and save/next flows.
- Build correction modal and Konva mask editor.
- Add single and bulk mask upload validation.
- Write review JSONL and versioned correction files.
- Recalculate accepted measurements.

Done when a reviewer can finish all 16 fields without editing files manually.

### Phase 4 — reports

- Build the aggregate Reports dashboard by scanning inspection manifests.
- Build the detailed cross-section report.
- Add printable HTML, CSV and JSON exports.
- Add explicit report approval.

Done when the displayed and exported KPIs match the accepted measurement JSON.

### Phase 5 — settings and model-improvement demonstration

- Validate and select a model path.
- Show correction-dataset counts and export.
- Add calibration-run and fine-tune-run controls as separate subprocesses.
- Display candidate metrics from validation JSON.
- Require an explicit promotion action.

Done when a candidate can be created and compared without modifying any completed inspection or existing prediction.

### Phase 6 — hardening and packaging

- Add input validation and helpful failure states.
- Add application/data-directory lock.
- Run keyboard and accessibility checks.
- Test Windows paths containing spaces.
- Wrap the working local web app in pywebview.
- Package with PyInstaller one-directory mode.

## 17. Prototype acceptance criteria

1. Importing a folder automatically starts inference for all images.
2. Review cannot begin until every valid field has a prediction or an explicit inference error.
3. The workspace supports both folder and table views.
4. Sample analysis shows the four primary business KPIs and a field risk map.
5. Field review shows original and prediction together, with synchronised navigation and display controls.
6. Accepting a mask records a review event and moves to the next field.
7. Selecting wrong opens the correction modal; edit/upload controls are not otherwise visible.
8. Saving a correction preserves the original prediction and creates a versioned mask.
9. Bulk upload previews filename, dimension and class-value errors before committing.
10. Sample KPIs recalculate from accepted masks.
11. Reports show disposition, reason, controlling field, complete KPI set and traceability.
12. The reports dashboard shows completed count and pass/fail/review percentages.
13. Settings validate a model before activation.
14. Reviewer feedback never changes the active model automatically.
15. All state survives an application restart by reading JSON and image artifacts from disk.
16. No database process or database file is created.

## 18. Minimum verification

### Frontend

- TypeScript strict compilation.
- One component test for field-status transitions.
- One mask-editor test covering brush, eraser and undo.
- One browser workflow: import fixture -> process -> correct one field -> complete review -> open report.

### Python/application layer

- One manifest round-trip test.
- One atomic-write recovery test.
- One mask validation test for dimensions and values.
- Golden measurement tests using known masks for areal fraction, count, Feret and cluster severity.
- One test proving a correction changes accepted measurements without changing prediction artifacts.

### Manual demonstration script

```text
1. Open Settings and validate the current production checkpoint.
2. Import the 32-image Test data set folder.
3. Show automatic batch progress.
4. Open the C/PEEK sample analysis and start priority review.
5. Accept one routine field.
6. Correct Field 12 or upload a corrected mask.
7. Show the sample KPI recalculation.
8. Complete/seed the remaining review decisions.
9. Open and export the cross-section report.
10. Open Model improvement and show the reviewed correction dataset.
```

## 19. Future value paths, not prototype scope

Once the core workflow is validated with reviewers, the same accepted-mask and measurement data can support:

- Material/batch trend charts and process-control limits.
- Supplier, process and material comparisons.
- Spatial defect heatmaps across a larger cross-section.
- Sampling-coverage and worst-field discovery.
- Microscope focus/exposure quality checks.
- Review-time and correction-rate savings.
- Model drift by material, microscope and calibration.
- Active-learning selection of the most informative fields.
- Integration with LIMS/QMS/MES systems.
- Multi-user review, permissions and immutable central audit storage.

These extensions should be added only after the local prototype proves that the measurements, prioritisation and report are useful in the real inspection workflow.

## 20. Implementation references

- [React with TypeScript](https://react.dev/learn/typescript)
- [Vite guide](https://vite.dev/guide/)
- [Konva React free drawing and erasing](https://konvajs.org/docs/react/Free_Drawing.html)
- [pywebview introduction](https://pywebview.flowrl.com/guide/)
- [FastAPI guidance for heavy background computation](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [PyTorch inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)

## 21. Design asset index

Use the following files as visual references, not as source data:

| Screen | Reference |
|---|---|
| Inspection workspace | [01-inspection-workspace.png](../../product-design/v2.1/01-inspection-workspace.png) |
| Sample analysis | [02-sample-analysis-kpis-v2.png](../../product-design/v2.1/02-sample-analysis-kpis-v2.png) |
| Field review | [03-field-review-kpis-v2.png](../../product-design/v2.1/03-field-review-kpis-v2.png) |
| Detailed report | [04-cross-section-quality-report-v2.png](../../product-design/v2.1/04-cross-section-quality-report-v2.png) |
| Model improvement | [05-model-improvement.png](../../product-design/v2.1/05-model-improvement.png) |
| Correction modal | [02-wrong-correction-modal.png](../../product-design/v1/02-wrong-correction-modal.png) |
| Reports dashboard | [03-final-reports.png](../../product-design/v1/03-final-reports.png) |
| Settings | [04-model-settings.png](../../product-design/v1/04-model-settings.png) |

The V2.1 visual corrections and KPI notes are documented in [product-design/v2.1/README.md](../../product-design/v2.1/README.md).
