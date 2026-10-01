"""Model and tokenizer loading.

Both models are treated identically: same head initialisation, same label mapping,
same sequence length. The *only* intended differences are the pre-trained encoder
and its tokenizer -- which is precisely the comparison the thesis is making.

Primary   : scherrmann/GermanFinBert_FP  -- German, financial-domain pre-training.
Comparison: ProsusAI/finbert             -- English, financial-sentiment pre-training.
"""
from __future__ import annotations

from typing import Any

from .config import MODEL_REGISTRY
from .utils import get_logger


def resolve_model_name(short_or_full: str) -> str:
    return MODEL_REGISTRY.get(short_or_full, short_or_full)


def load_model_and_tokenizer(
    name: str,
    num_labels: int,
    id2label: dict[int, str] | None = None,
) -> tuple[Any, Any, dict]:
    """Load a fresh sequence-classification model with a randomly initialised head."""
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    log = get_logger()
    hf_name = resolve_model_name(name)

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    conf = AutoConfig.from_pretrained(hf_name, num_labels=num_labels)
    if id2label:
        conf.id2label = {int(k): str(v) for k, v in id2label.items()}
        conf.label2id = {str(v): int(k) for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name, config=conf, ignore_mismatched_sizes=True
    )

    info = {
        "short_name": name,
        "hf_name": hf_name,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "do_lower_case": bool(getattr(tokenizer, "do_lower_case", False)),
        "model_class": type(model).__name__,
        "num_labels": int(num_labels),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "max_position_embeddings": int(getattr(conf, "max_position_embeddings", -1)),
    }
    log.info(
        "Loaded %s (%s) | vocab %d | lower_case=%s | %.1fM params",
        name, hf_name, info["vocab_size"], info["do_lower_case"],
        info["n_parameters"] / 1e6,
    )
    return model, tokenizer, info


def tokenizer_diagnostics(tokenizer: Any, texts: list[str], max_length: int) -> dict:
    """How well does this tokenizer cope with German transaction text?

    Reported per model so the German-vs-English vocabulary difference is visible as
    a measured quantity rather than an assertion.
    """
    sample = texts[:2000]
    enc = tokenizer(sample, truncation=True, max_length=max_length)
    lengths = [len(x) for x in enc["input_ids"]]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    n_unk = 0
    if unk_id is not None:
        n_unk = sum(ids.count(unk_id) for ids in enc["input_ids"])
    n_words = sum(len(t.split()) for t in sample)
    n_subtokens = sum(lengths) - 2 * len(sample)  # minus [CLS]/[SEP]
    return {
        "n_texts": len(sample),
        "mean_tokens": round(sum(lengths) / max(len(lengths), 1), 2),
        "max_tokens": max(lengths) if lengths else 0,
        "p95_tokens": int(sorted(lengths)[int(0.95 * len(lengths))]) if lengths else 0,
        "n_truncated": int(sum(1 for x in lengths if x >= max_length)),
        "subtokens_per_word": round(n_subtokens / max(n_words, 1), 3),
        "n_unk_tokens": int(n_unk),
    }
