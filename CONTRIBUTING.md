# Contributing

1. Branch from `main`; small, descriptive commits.
2. Install CPU PyTorch, then `pip install -r requirements-dev.txt`.
3. Before pushing: `ruff check .`, `pytest`, and for research changes
   `python verify_dataset.py && python verify_augmentation.py && python final_evaluation.py`.
4. Research rules (see `AUDIT.md`, `FINAL_REPORT.md`):
   * never recompute the frozen split or corpus; scripts verify their checksums;
   * never use the test split for training, selection, calibration or retraining;
   * keep the 0.8 threshold unless a documented, evidence-based decision changes it;
   * do not overwrite historical results; add new runs under new names;
   * the research files store CRLF/LF bytes exactly (`.gitattributes: * -text`); keep
     line endings unchanged when editing them.
5. Privacy: never log or persist transaction descriptions; do not add them to tests' expected output.
6. No secrets in code or config; `.env` is git-ignored.
7. Update `CHANGELOG.md` and the relevant documents; CI must pass.
