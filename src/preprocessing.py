"""German-aware preprocessing of bank transaction descriptions.

Design principle (from the brief): *be conservative*. Remove only tokens that are
provably non-informative for the category -- dates and reference/invoice numbers --
and preserve vendor names, financial terms and meaningful alphanumerics.

Every transformation is numbered and documented in `TRANSFORMATIONS`, and
`clean_series` returns a per-step audit trail so the effect can be reported.

Difference to the prototype
---------------------------
The prototype's cleaner applied ``re.sub(r"[^a-zaeoeuess\\s]", " ", text)``, which
deletes *every* digit unconditionally. That turned ``B2B`` into the bigram ``b b``,
a token then shared by three unrelated categories. This module protects such tokens
explicitly (see `PreprocessingConfig.protected_tokens`).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

import pandas as pd

from .config import PreprocessingConfig

TRANSFORMATIONS = [
    "T0  Unicode NFKC normalisation and whitespace strip",
    "T1  Protect configured alphanumeric tokens (B2B, 365, Q1 ...) behind placeholders",
    "T2  Remove ISO-like dates: 2024-06, 2024/06, 2024.06.15",
    "T3  Remove German dates: 01.06.2024, 1/6/24",
    "T4  Remove IBAN-like tokens (2 letters + 2 digits + >=10 alphanumerics)",
    "T5  Remove invoice/reference codes: RE-2024-8096, INV12345, ABC-99887",
    "T6  Remove standalone digit runs of length >= 3 (account/reference fragments)",
    "T7  Replace remaining punctuation with a space; keep letters, umlauts, digits",
    "T8  Expand German banking abbreviations (ueberw -> ueberweisung, ...)",
    "T9  (optional) Remove legal forms (GmbH, AG, KG ...)         [OFF by default]",
    "T10 (optional) Remove domain stop words (ref, beleg, months) [OFF by default]",
    "T11 Restore protected tokens",
    "T12 Lower-case (configurable)",
    "T13 Collapse whitespace; substitute placeholder if the result is empty",
]

# --- T2/T3: dates -----------------------------------------------------------
_RE_ISO_DATE = re.compile(r"\b(?:19|20)\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?\b")
_RE_DE_DATE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./](?:19|20)?\d{2}\b")

# --- T4 ---------------------------------------------------------------------
_RE_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,}\b", re.IGNORECASE)

# --- T5: invoice / reference codes ------------------------------------------
# RE-2024-8096, RG-2024-1, INV12345, ABC-99887, 2024-8096
_RE_REF_NUMERIC = re.compile(r"\b\d{4}-\d{2,}\b")
_RE_REF_HYPHEN = re.compile(r"\b[A-Za-zÄÖÜäöü]{1,5}-\d{2,}(?:-\d{2,})*\b")
_RE_REF_COMPACT = re.compile(r"\b[A-Za-zÄÖÜäöü]{2,5}\d{4,}\b")

# --- T6 ---------------------------------------------------------------------
_RE_LONG_DIGITS = re.compile(r"\b\d{3,}\b")

# --- T7 ---------------------------------------------------------------------
# Keep latin letters (incl. umlauts / accents), digits and whitespace.
_RE_PUNCT = re.compile(r"[^0-9A-Za-zÀ-ɏ\s]")
_RE_WS = re.compile(r"\s+")

# --- T8: German banking abbreviations ---------------------------------------
# Deliberately small and unambiguous. Each entry *adds* information rather than
# removing it, which is why this step is ON by default.
ABBREVIATIONS: dict[str, str] = {
    "ueberw": "ueberweisung",
    "überw": "überweisung",
    "uebw": "ueberweisung",
    "lastschr": "lastschrift",
    "lastschrft": "lastschrift",
    "gutschr": "gutschrift",
    "dauerauftr": "dauerauftrag",
    "vwz": "verwendungszweck",
    "rg": "rechnung",
    "rechn": "rechnung",
    "sozialvers": "sozialversicherung",
    "versich": "versicherung",
    "abschl": "abschlag",
    "instandh": "instandhaltung",
    "wartungsvertr": "wartungsvertrag",
    "mwst": "umsatzsteuer",
    "ust": "umsatzsteuer",
    "zahlg": "zahlung",
}

# --- T9 ---------------------------------------------------------------------
LEGAL_FORMS = {
    "gmbh", "mbh", "ag", "kg", "ohg", "ug", "gbr", "kgaa", "se", "ek", "eg",
    "ltd", "llc", "inc", "plc", "bv", "nv", "sa", "srl", "spa", "cie",
}

# --- T10 --------------------------------------------------------------------
DOMAIN_STOPWORDS = {
    "ref", "referenz", "beleg", "belegnr", "nr", "nummer", "konto", "kto",
    "iban", "bic", "blz", "mandat", "mandatsref", "kundenreferenz",
    "jan", "feb", "mrz", "maerz", "apr", "mai", "jun", "jul", "aug",
    "sep", "sept", "okt", "nov", "dez",
}

# Placeholder marker for T1. Pure ASCII letters, so it survives T2-T7 untouched. It
# must not collide with real German vocabulary nor with any dictionary entry above.
_MARK = "qxzprot"


def _index_to_letters(i: int) -> str:
    """0 -> 'a', 1 -> 'b', ..., 26 -> 'aa'. Letters only, so T7 cannot damage it."""
    out = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def _protect(text: str, tokens: Iterable[str]) -> tuple[str, dict[str, str]]:
    """Replace protected tokens with letter-only placeholders (T1).

    Longest tokens first, so that e.g. ``24/7`` is matched before ``24``.
    """
    mapping: dict[str, str] = {}
    boundary_l = r"(?<![0-9A-Za-zÀ-ɏ])"
    boundary_r = r"(?![0-9A-Za-zÀ-ɏ])"
    for i, tok in enumerate(sorted(set(tokens), key=len, reverse=True)):
        if not tok:
            continue
        pattern = re.compile(boundary_l + re.escape(tok) + boundary_r, re.IGNORECASE)
        if pattern.search(text):
            placeholder = f"{_MARK}{_index_to_letters(i)}{_MARK}"
            mapping[placeholder] = tok
            text = pattern.sub(placeholder, text)
    return text, mapping


def _restore(text: str, mapping: dict[str, str]) -> str:
    for placeholder, tok in mapping.items():
        text = text.replace(placeholder, tok)
    return text


def clean_description(text: object, cfg: PreprocessingConfig | None = None) -> str:
    """Apply T0-T13 to a single description. Pure function, no global state."""
    cfg = cfg or PreprocessingConfig()

    # T0
    s = unicodedata.normalize("NFKC", str(text)).strip()

    # T1
    s, protected = _protect(s, cfg.protected_tokens)

    # T2 / T3
    if cfg.strip_dates:
        s = _RE_ISO_DATE.sub(" ", s)
        s = _RE_DE_DATE.sub(" ", s)

    # T4 -- before T5/T6 so IBANs are not partially consumed
    s = _RE_IBAN.sub(" ", s)

    # T5 -- order matters: the prefixed form must be matched first. Removing the
    # numeric tail of "RE-2024-6278" first would strand the orphan token "re".
    if cfg.strip_reference_numbers:
        s = _RE_REF_HYPHEN.sub(" ", s)
        s = _RE_REF_NUMERIC.sub(" ", s)
        s = _RE_REF_COMPACT.sub(" ", s)

    # T6
    if cfg.strip_long_digit_runs:
        s = _RE_LONG_DIGITS.sub(" ", s)

    # T7
    s = _RE_PUNCT.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()

    # T8 / T9 / T10 operate token-wise
    if cfg.expand_abbreviations or cfg.remove_legal_forms or cfg.remove_domain_stopwords:
        out: list[str] = []
        for tok in s.split():
            if _MARK in tok:  # protected placeholder -- never touched
                out.append(tok)
                continue
            low = tok.lower().rstrip(".")
            if cfg.remove_legal_forms and low in LEGAL_FORMS:
                continue
            if cfg.remove_domain_stopwords and low in DOMAIN_STOPWORDS:
                continue
            if cfg.expand_abbreviations and low in ABBREVIATIONS:
                out.append(ABBREVIATIONS[low])
                continue
            out.append(tok)
        s = " ".join(out)

    # T11
    s = _restore(s, protected)

    # T12
    if cfg.lowercase:
        s = s.lower()

    # T13
    s = _RE_WS.sub(" ", s).strip()
    return s if s else cfg.empty_placeholder


# --------------------------------------------------------------------------- #
#  Legacy cleaner -- provenance resolution only
# --------------------------------------------------------------------------- #
_LEGACY_DATE = re.compile(r"\b20\d{2}[-/.]\d{1,2}\b")
_LEGACY_REF = re.compile(r"\b[a-z]{1,4}-?\d{3,}\b")
_LEGACY_KEEP = re.compile(r"[^a-zäöüß\s]")


def legacy_clean_description(text: object) -> str:
    """The prototype notebook's ``clean_description``, reproduced verbatim.

    This is **not** used to prepare model input -- it is strictly worse than the
    cleaner above (it destroys ``B2B`` into ``b b``). It exists for exactly one
    reason: the shipped synthetic corpus records its ``source_text`` provenance in
    *this* normalisation, so resolving which real pattern a synthetic example came
    from requires reproducing it. See `splitting.filter_synthetic_for_train`.
    """
    t = str(text).lower().strip()
    t = _LEGACY_DATE.sub(" ", t)
    t = _LEGACY_REF.sub(" ", t)
    t = _LEGACY_KEEP.sub(" ", t)
    return _RE_WS.sub(" ", t).strip()


def clean_series(
    texts: pd.Series, cfg: PreprocessingConfig | None = None
) -> tuple[pd.Series, dict]:
    """Vectorised cleaning plus an audit trail for the documentation."""
    cfg = cfg or PreprocessingConfig()
    raw = texts.astype(str)
    cleaned = raw.map(lambda t: clean_description(t, cfg))

    raw_tokens = Counter(t for s in raw for t in s.split())
    new_tokens = Counter(t for s in cleaned for t in s.split())
    # Compare case-insensitively when lower-casing is enabled, otherwise a token that
    # was merely re-cased ("SaaS" -> "saas") would be miscounted as deleted.
    fold = (lambda t: t.lower()) if cfg.lowercase else (lambda t: t)
    surviving = {fold(t) for t in new_tokens}
    removed = Counter({t: c for t, c in raw_tokens.items() if fold(t) not in surviving})

    audit = {
        "transformations_applied": TRANSFORMATIONS,
        "config": dict(cfg.__dict__),
        "n_texts": int(len(raw)),
        "unique_raw": int(raw.nunique()),
        "unique_cleaned": int(cleaned.nunique()),
        "collapse_ratio": round(float(raw.nunique()) / max(int(cleaned.nunique()), 1), 3),
        "empty_after_clean": int((cleaned == cfg.empty_placeholder).sum()),
        "mean_tokens_raw": round(float(raw.map(lambda s: len(s.split())).mean()), 3),
        "mean_tokens_cleaned": round(float(cleaned.map(lambda s: len(s.split())).mean()), 3),
        "vocab_raw": len(raw_tokens),
        "vocab_cleaned": len(new_tokens),
        "n_token_types_removed": len(removed),
        "removed_token_types_note": (
            "A token type counts as removed only if no case-folded form of it survives "
            "cleaning; pure re-casing is not counted as removal."
        ),
        "most_removed_token_types": removed.most_common(30),
    }
    return cleaned, audit


def sample_transformations(
    texts: pd.Series, cfg: PreprocessingConfig | None = None, n: int = 25
) -> pd.DataFrame:
    """Before/after table for the documentation."""
    cfg = cfg or PreprocessingConfig()
    take = pd.Series(texts.astype(str).unique()).head(n)
    return pd.DataFrame({"raw": take, "cleaned": [clean_description(t, cfg) for t in take]})
