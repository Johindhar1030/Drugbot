"""Unicode script and language detector for DrugBot multilingual PDF ingestion and RAG.

Detects languages across Latin, South Asian (Tamil, Hindi, Malayalam, Telugu, Kannada, Bengali, Gujarati),
East Asian (Chinese, Japanese, Korean), Middle Eastern (Arabic), and Cyrillic scripts without external dependencies.
"""
import re
from typing import Set

# Common Latin stop word sets for distinguishing major European languages
_LATIN_STOPWORDS = {
    "en": {"the", "and", "for", "that", "this", "with", "from", "have", "not", "were", "which", "been", "is"},
    "es": {"el", "la", "los", "las", "un", "una", "del", "por", "para", "con", "como", "mas"},
    "fr": {"le", "la", "les", "un", "une", "des", "du", "et", "dans", "pour", "avec", "est"},
    "de": {"der", "die", "das", "und", "in", "den", "von", "mit", "ist", "des", "nicht", "eine"},
    "it": {"il", "la", "che", "del", "con", "per", "non", "uno", "una", "dei", "sono"},
    "pt": {"com", "para", "como", "mais", "pela", "pelo", "uma", "este", "esta"},
}


def detect_language(text: str) -> str:
    """Detect primary ISO 639-1 language code from string using script ranges & character distribution.
    
    Supports: en, ta, hi, ml, te, kn, bn, gu, zh, ja, ko, ar, ru, fr, de, es, it, pt, or 'unknown'.
    """
    if not text or len(text.strip()) == 0:
        return "unknown"

    script_counts = {
        "ta": 0,  # Tamil
        "hi": 0,  # Hindi (Devanagari)
        "ml": 0,  # Malayalam
        "te": 0,  # Telugu
        "kn": 0,  # Kannada
        "bn": 0,  # Bengali
        "gu": 0,  # Gujarati
        "zh": 0,  # Chinese
        "ja": 0,  # Japanese
        "ko": 0,  # Korean
        "ar": 0,  # Arabic
        "ru": 0,  # Russian / Cyrillic
        "latin": 0,
    }

    for char in text:
        cp = ord(char)
        if 0x0B80 <= cp <= 0x0BFF:
            script_counts["ta"] += 1
        elif 0x0900 <= cp <= 0x097F:
            script_counts["hi"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            script_counts["ml"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            script_counts["te"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            script_counts["kn"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            script_counts["bn"] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            script_counts["gu"] += 1
        elif (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF):
            script_counts["ja"] += 1
        elif (0xAC00 <= cp <= 0xD7AF) or (0x1100 <= cp <= 0x11FF):
            script_counts["ko"] += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            script_counts["zh"] += 1
        elif (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F):
            script_counts["ar"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            script_counts["ru"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x00FF):
            script_counts["latin"] += 1

    # Find maximum non-latin script count
    non_latin = {k: v for k, v in script_counts.items() if k != "latin"}
    max_non_latin_lang = max(non_latin, key=non_latin.get)
    max_non_latin_count = non_latin[max_non_latin_lang]

    if max_non_latin_count >= 3:
        return max_non_latin_lang

    if script_counts["latin"] > 0:
        words = set(re.findall(r"\b[a-z]{2,}\b", text.lower()))
        best_lang = "en"
        best_overlap = 0
        for lang_code, stop_set in _LATIN_STOPWORDS.items():
            overlap = len(words & stop_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_lang = lang_code
        return best_lang

    return "unknown"


def detect_multilingual(text: str) -> str:
    """Return comma-separated string of unique detected languages in text (e.g. 'en,ta')."""
    if not text:
        return "unknown"
    langs = []
    primary = detect_language(text)
    if primary != "unknown":
        langs.append(primary)

    # Check secondary scripts
    for lang in ["en", "ta", "hi", "ml", "te", "kn", "bn", "gu", "zh", "ja", "ko", "ar", "ru"]:
        if lang not in langs and _has_script(text, lang):
            langs.append(lang)

    return ",".join(langs) if langs else "unknown"


def _has_script(text: str, target_lang: str) -> bool:
    count = 0
    for char in text:
        cp = ord(char)
        if target_lang == "ta" and 0x0B80 <= cp <= 0x0BFF:
            count += 1
        elif target_lang == "hi" and 0x0900 <= cp <= 0x097F:
            count += 1
        elif target_lang == "zh" and 0x4E00 <= cp <= 0x9FFF:
            count += 1
        elif target_lang == "ja" and ((0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF)):
            count += 1
        elif target_lang == "ko" and ((0xAC00 <= cp <= 0xD7AF) or (0x1100 <= cp <= 0x11FF)):
            count += 1
        elif target_lang == "ar" and 0x0600 <= cp <= 0x06FF:
            count += 1
        elif target_lang == "ru" and 0x0400 <= cp <= 0x04FF:
            count += 1
        elif target_lang == "en" and ((0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A)):
            count += 1
    return count >= 5
