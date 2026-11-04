#!/usr/bin/env python3
"""Record sentence-transformers (torch) embeddings as the reference for the ONNX encoder.

The app no longer depends on torch. This one-off script needs it, so run it with the reference
stack pulled in for this command only:

    uv run --with "sentence-transformers==5.2.0" python scripts/make_embedding_reference.py

It writes tests/fixtures/embeddings/minilm_reference.json: the torch embeddings of a fixed list
of synthetic clinical phrases, and the torch top-k golden-example matches for each phrase.
tests/core/test_embeddings_equivalence.py compares the ONNX encoder against it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from clinical_analytics.core.embeddings import MODEL_NAME, MODEL_REPO, MODEL_REVISION  # noqa: E402

OUT = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "minilm_reference.json"
TOP_K = 3

# Synthetic phrases only. Edge cases: empty, single token, punctuation, numbers, non-ASCII,
# casing, and one phrase longer than the 256-token limit (exercises truncation).
PHRASES = [
    "how many patients?",
    "how many patients by status?",
    "what is the average age?",
    "compare LDL between treatment groups",
    "what predicts mortality?",
    "is BMI correlated with HDL cholesterol?",
    "show the distribution of systolic blood pressure",
    "how many diabetics are on statins",
    "average follow-up days for patients who had an event",
    "patients older than 65 with hypertension",
    "survival by treatment arm",
    "DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis",
    "Statin use 1: Yes 2: No",
    "excluding missing values, what is the median creatinine?",
    "which site enrolled the most women?",
    "odds ratio of readmission for smokers vs non-smokers",
    "HbA1c > 7.5%",
    "Évaluation de la pression artérielle chez les patients âgés",
    "WHAT IS THE MEAN EGFR",
    "remove those with n/a",
    "",
    "age",
    "?!",
    "what about by sex",
    "now only for the drug group",
    "count of events per 1000 patient-years",
    " ".join(["the patient reported chest pain and shortness of breath"] * 40),
]


def main() -> int:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from clinical_analytics.core.config_loader import load_golden_examples_config

    model = SentenceTransformer(f"{MODEL_REPO}", revision=MODEL_REVISION, device="cpu")
    phrase_emb = model.encode(PHRASES, convert_to_numpy=True, normalize_embeddings=False)
    golden = [q["query"] for q in load_golden_examples_config()["questions"]]
    golden_emb = model.encode(golden, convert_to_numpy=True)
    sims = phrase_emb @ golden_emb.T
    top_k = np.argsort(-sims, axis=1)[:, :TOP_K]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL_NAME,
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "generator": f"sentence-transformers {__import__('sentence_transformers').__version__}, "
        f"torch {__import__('torch').__version__}, cpu",
        "top_k": TOP_K,
        "golden_example_count": len(golden),
        "phrases": PHRASES,
        "embeddings": [[round(float(x), 7) for x in row] for row in phrase_emb],
        "golden_top_k": top_k.tolist(),
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {len(PHRASES)} phrases x {phrase_emb.shape[1]} dims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
