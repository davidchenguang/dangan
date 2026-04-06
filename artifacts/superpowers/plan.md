# Implementation Plan - Fix <image> Token Mismatch & Unified OCR Cleaning

The project currently has 4 failing tests due to a mismatch between implementation requirements (DeepSeek-OCR-2 needing `<image>` in prompts) and outdated test assertions. Additionally, OCR output may occasionally contain these system tokens, necessitating a unified cleaning layer.

## User Review Required

> [!IMPORTANT]
> The prompt templates in `core/models.py` were updated to include `<image>` because the DeepSeek-OCR-2 model requires it for token insertion. The tests were left outdated, asserting that `<image>` is *not* present. I will update the tests to reflect the new requirement.

> [!NOTE]
> I will also add a unified cleaning step in the OCR engine to ensure that if the model returns the `<image>` tag in its text output, it is stripped before being passed to downstream logic.

## Proposed Changes

---

### Core OCR Processing

#### [MODIFY] [ocr_engine.py](file:///d:/dangan/core/ocr_engine.py)
- Update `_extract_result` to include a regex-based cleaning step for the final text.
- Remove `<image>`, `<|grounding|>`, and other `<|...|>` tags from the output.

---

### Tests

#### [MODIFY] [test_models.py](file:///d:/dangan/tests/test_models.py)
- Update `test_verbatim_prompt`, `test_structured_prompt`, and `test_json_prompt` to assert that `<image>` **is** present in the prompts (matching `core/models.py`).

#### [MODIFY] [test_config.py](file:///d:/dangan/tests/test_config.py)
- Update `test_ocr_defaults` to expect `"verbatim"` instead of `"structured"`, as `"verbatim"` is the current default in `config.yaml`.

## Open Questions

- None at this time. The failures are clearly due to out-of-sync tests and the lack of an output cleaning layer.

## Verification Plan

### Automated Tests
- Run the full test suite using the virtual environment:
  ```powershell
  .\.venv\Scripts\pytest -v
  ```
- Expected result: 115 passed, 0 failed.

### Manual Verification
- None required as this is a fix for automated tests and internal consistency.

