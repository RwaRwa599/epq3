# Block 1 — Document Normalization & ROI Extraction

`new2` is a modular medical lab-request document pipeline. This branch implements **Block 1 only**: take a raw photo or scan of a laboratory request sheet and emit a rectified canonical canvas plus illumination-normalized field crops.

Do not add agent-memory files (`memory/`, `agent-skills/`, `AGENTS.md`, protocol docs, hooks, or agent-memory CI) to this repository or to new block packages.

## Pipeline

1. Detect the page quadrilateral and warp it to the canonical canvas (`2048×1754`).
2. Correct 90/180/270 orientation so the printed header sits at the top.
3. Fine-align using header/footer landmarks and column checkbox gutters.
4. Crop every checkbox and handwriting ROI from relative `[0, 1]` template coordinates.
5. Flatten illumination (background division + CLAHE) and attach a quality score.

Downstream blocks (mark classification, HTR, knowledge graph, HiTL) consume `NormalizedDocumentResult`.

## Usage

```python
from med_doc import normalize_document

result = normalize_document("sheet.jpg")
tick = result.checkbox_crops["body_check_plan_1"]
print(tick.canonical_bbox, tick.quality_score)
```

## Layout

- `src/med_doc/schemas.py` — `FieldCrop`, `TemplateSpec`, `NormalizedDocumentResult`
- `src/med_doc/normalization/` — warp, align, crops, viz, pipeline
- `templates/lab_request_canonical.json` — relative-coordinate lab-request template (from clinic print v1)
- `tests/test_normalization.py`

## Local checks

```bash
pip install -e ".[dev]"
pytest
```

Private / unredacted clinic photos belong in `data/samples/private/` (gitignored). Never commit PHI.
