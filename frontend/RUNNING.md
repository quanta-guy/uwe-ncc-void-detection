# Running the prototype

One command, from the repo root:

```powershell
.\demo.ps1          # starts everything, opens the browser; Ctrl+C stops it all
.\demo.ps1 -Stop    # kills anything left on the demo ports
```

It builds the fixtures if they are missing, starts both servers, waits until each
actually answers, reports whether Ollama is up, and cleans up its own processes on
exit.

Manually, it is three processes. The first two are required; the third only powers
the report assistant.

```bash
# 1. build the preloaded samples (once, ~2 min on GPU)
conda run -n uwe_hack python frontend/tools/build_fixtures.py

# 2. inference backend - weight loading, import, live prediction
cd frontend && npm run api            # http://127.0.0.1:8000

# 3. the app
cd frontend && npm run dev            # http://127.0.0.1:5173

# optional: report assistant
ollama serve && ollama pull qwen3:4b
```

## What needs which process

| Screen | Works without the backend? |
|---|---|
| Inspections, sample analysis, field review, reports | Yes — reads `fixtures.json` |
| **New inspection → import and run inference** | **No** |
| **Settings → checkpoint list, Validate and activate** | **No** |
| Reports → AI assistant | Needs Ollama, not the backend |

The app deliberately still opens when the backend is down: the two preloaded Test
samples render read-only, and the screens that need live inference say so instead of
failing silently. That split is why a missing backend is a message, not an error page.

## Importing

`Folder` takes a whole directory (`webkitdirectory`); `Images` takes a selection.
Calibration is mandatory — without a trusted µm/pixel there is no physical
measurement, only pixel counts.

Imported inspections are written to `public/data/imported.json` and merged at load.
`fixtures.json` is never touched, so the preloaded samples stay reproducible from
`build_fixtures.py`. `DELETE /api/inspections/{id}` removes an import and its render
layers.

## Swapping the model

Settings discovers every `*.pt` under `runs/` (the submission model) and `models/`
(the curated alternatives), grouping folds into the ensembles they form. `Validate
and activate` opens the checkpoint, pushes a real tensor through it, and only
activates it if the output is three classes at the input's own size. Selecting a
model also shows its validation record — an out-of-fold confusion matrix for the two
ensembles that were measured that way, and an honest "no matrix was measured" for the
singles.

`archive/` is deliberately not scanned: solution 2/3 checkpoints there need their own
preprocessing (normalisation, physical resampling) that this pipeline does not apply.
The shape gate would pass them and they would then quietly mispredict.

## Tests

```bash
cd frontend
npm test                        # visual (8 routes) + interaction (11) + import (12)
node tools/chat_test.mjs        # report assistant grounding; needs Ollama
conda run -n uwe_hack python server/app.py --demo
conda run -n uwe_hack python tools/severity_geometry.py
```

`import_test.mjs` needs both servers up. It drives the real UI rather than the API,
because an endpoint that works behind a dead button is still a broken product.
