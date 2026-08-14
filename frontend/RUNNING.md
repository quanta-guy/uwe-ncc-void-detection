# Running the prototype

Three processes. The first two are required; the third only powers the report
assistant.

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

Settings discovers every `*.pt` under `runs/`, `archive/checkpoints/` and
`solution*/runs/`, grouping folds into the ensembles they form. `Validate and
activate` opens the checkpoint, pushes a real tensor through it, and only activates it
if the output is three classes at the input's own size. Checkpoints from solutions 2
and 3 are listed and will fail that gate where their builder differs — which is the
gate doing its job, not a bug.

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
