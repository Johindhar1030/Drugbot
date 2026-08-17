"""Controlled Query Normalization & Typo Handling for DrugBot.

Provides a robust, medical-aware normalization layer that executes BEFORE retrieval.
- Deterministic cleanup (whitespace, punctuation, grammatical agreements)
- Strict protected entity preservation (drug names, CYP enzymes, dosages, units, section numbers, medical abbreviations)
- Medical dictionary-driven typo correction with conservative Levenshtein edit distance & tie-breaking
- Ambiguity and incomplete short-query detection with polite clarification generation
- Zero unnecessary LLM calls (100% fast, deterministic execution)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.vector_store import get_known_drugs

logger = logging.getLogger(__name__)


@dataclass
class CorrectionDetail:
    original: str
    replacement: str
    confidence: float
    reason: str = "typo_correction"

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "replacement": self.replacement,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


@dataclass
class NormalizationResult:
    original_query: str
    normalized_query: str
    changed: bool
    confidence: float
    corrections: list[CorrectionDetail] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_message: str | None = None
    protected_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "changed": self.changed,
            "confidence": round(self.confidence, 4),
            "corrections": [c.to_dict() for c in self.corrections],
            "needs_clarification": self.needs_clarification,
            "clarification_message": self.clarification_message,
            "protected_entities": self.protected_entities,
        }


# ── Curated Medical Terminology Dictionary ─────────────────────────────────

# Common FDA prescribing information section terms and concepts
_BASE_MEDICAL_DICTIONARY: set[str] = {
    # Section headings & regulatory categories
    "indication", "indications", "usage",
    "dosage", "dosages", "dosing", "administration",
    "contraindication", "contraindications",
    "warning", "warnings", "precaution", "precautions",
    "adverse", "reaction", "reactions", "side", "effect", "effects",
    "interaction", "interactions",
    "use", "specific", "populations", "population",
    "pediatric", "geriatric", "pregnancy", "lactation", "nursing",
    "renal", "hepatic", "impairment", "dialysis",
    "overdosage", "overdose",
    "description", "ingredients", "composition",
    "clinical", "pharmacology", "pharmacokinetics", "pharmacodynamics",
    "nonclinical", "toxicology", "carcinogenesis", "mutagenesis", "fertility",
    "studies", "study", "trial", "trials",
    "storage", "handling", "supplied",
    "patient", "counseling", "information", "medication", "guide",
    "boxed", "warning", "highlights", "prescribing", "package", "insert",

    # Clinical conditions, diseases & indications
    "rheumatoid", "arthritis",
    "psoriatic",
    "ankylosing", "spondylitis",
    "spondyloarthritis", "axial",
    "crohn", "crohn's", "crohns", "disease",
    "ulcerative", "colitis",
    "plaque", "psoriasis",
    "hidradenitis", "suppurativa",
    "uveitis",
    "atopic", "dermatitis", "eczema",
    "juvenile", "idiopathic",
    "arteritis", "takayasu",
    "lupus", "erythematosus",
    "asthma", "sinusitis", "polyps",
    "infection", "infections", "tuberculosis", "latent", "active",
    "malignancy", "malignancies", "lymphoma", "melanoma",
    "thrombosis", "embolism", "deep", "vein", "pulmonary",
    "hypersensitivity", "anaphylaxis", "angioedema", "urticaria",
    "immunosuppression", "immunosuppressive", "immunosuppressant",
    "gastrointestinal", "perforation", "diverticulitis",
    "hepatotoxicity", "reactivation", "hepatitis",
    "cardiovascular", "mortality", "myocardial", "infarction", "stroke",
    "neutropenia", "lymphopenia", "anemia", "cytopenia",
    "lipids", "cholesterol", "triglycerides", "transaminases",
    "shingles", "zoster", "herpes", "varicella",

    # Pharmacology, regimens & dosing terms
    "recommended", "initial", "starting", "maintenance", "induction",
    "monotherapy", "combination", "concomitant", "concomitantly",
    "titration", "discontinuation", "interruption", "reduction",
    "bioavailability", "clearance", "half-life", "excretion", "metabolism",
    "absorption", "distribution", "steady-state", "steady", "state",
    "inhibitor", "inhibitors", "inducer", "inducers", "substrate", "substrates",
    "potent", "strong", "moderate", "weak",
    "subcutaneous", "subcutaneously", "oral", "orally", "intravenous", "intravenously",
    "injection", "injectable", "autoinjector", "prefilled", "syringe", "pen",
    "vial", "tablet", "tablets", "capsule", "capsules", "solution", "extended-release",
    "daily", "weekly", "monthly", "every", "other", "week",
    "milligram", "milligrams", "microgram", "micrograms", "milliliter", "milliliters",
    "effectiveness", "efficacy", "safety", "tolerability", "remission", "response",
    "baseline", "screening", "monitoring", "laboratory", "evaluations", "tests",
}

# Protected medical acronyms & symbols (always preserve exact casing / no typo alteration)
_PROTECTED_ACRONYMS: set[str] = {
    "RA", "PSA", "AS", "CD", "UC", "PSO", "HS", "JIA", "AD", "GCA", "TAK",
    "JAK", "JAK1", "JAK2", "JAK3", "TYK2", "TNF", "TNFA", "IL-12", "IL-23", "IL-17", "IL-4", "IL-13", "IL-1", "IL-6",
    "CYP3A4", "CYP2C19", "CYP2D6", "CYP1A2", "CYP2C9", "CYP3A", "CYP2C", "CYP2D", "CYP1A",
    "IV", "SC", "PO", "IM", "BID", "QD", "TID", "QID", "PRN", "Q2W", "Q4W", "Q8W", "Q12W",
    "MTX", "DMARD", "CSDMARD", "BDMARD", "TSDMARD", "NSAID", "NSAIDS",
    "TB", "LTBI", "HBV", "HCV", "HIV", "CBC", "ALC", "ANC", "ALT", "AST", "ULN",
    "EGFR", "GFR", "CRCL", "DVT", "PE", "VTE", "MACE", "GI", "FDA", "PI", "USPI",
}

# Known common drug brand and generic names
_KNOWN_DRUG_NAMES: set[str] = {
    "RINVOQ", "HUMIRA", "SKYRIZI", "DUPIXENT", "KEYTRUDA", "OPDIVO", "STELARA", "REMICADE",
    "CIBINQO", "XELJANZ", "OLUMIANT", "COSENTYX", "TALTZ", "TREMFYA", "ILUMYA", "SIMPONI",
    "ENBREL", "CIMZIA", "ENTYVIO", "OTEZLA", "SILIQ", "ACTEMRA", "KEVZARA", "BIMZELX",
    # Generics
    "UPADACITINIB", "ADALIMUMAB", "RISANKIZUMAB", "DUPILUMAB", "PEMBROLIZUMAB",
    "NIVOLUMAB", "USTEKINUMAB", "INFLIXIMAB", "ABROCITINIB", "TOFACITINIB",
    "BARICITINIB", "SECUKINUMAB", "IXEKIZUMAB", "GUSELKUMAB", "TILDRAKIZUMAB",
    "GOLIMUMAB", "ETANERCEPT", "CERTOLIZUMAB", "VEDOLIZUMAB", "APREMILAST",
    "BRODALUMAB", "TOCILIZUMAB", "SARILUMAB", "BIMEKIZUMAB",
}

# Common standard English words that should NEVER be replaced as typos
_STOPWORDS_AND_COMMON_WORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doesn't", "doing", "don't", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "isn't", "it", "its", "itself", "let", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    "tell", "give", "list", "show", "describe", "explain", "compare", "find",
    "table", "recommended", "missed", "skip", "take", "taking", "stop", "start",
    "prescribed", "people", "adult", "adults", "children", "patients", "doctor",
}


# ── Protected Entity Regex Matchers ────────────────────────────────────────

# Matches CYP enzymes (e.g. CYP3A4, CYP2C19, CYP2D6)
_CYP_PATTERN = re.compile(r"\bCYP[0-9][A-Z][0-9]{1,2}(?:/[0-9]+)?\b", re.IGNORECASE)

# Matches dosages and units (e.g. 15 mg, 30mg, 45 mg/mL, 0.8 mL, 40mg/0.8ml, 100 mcg, 20%)
_DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mg/ml|mg/\d+(?:\.\d+)?\s*ml|units?|tablets?|capsules?|%)\b",
    re.IGNORECASE,
)

# Matches frequency & regimen expressions (e.g. once daily, twice daily, every other week)
_REGIMEN_PATTERN = re.compile(
    r"\b(?:once|twice|thrice|1 time|2 times|3 times)\s+daily\b|"
    r"\b(?:every|q)\s*(?:other\s+week|\d+\s*(?:weeks?|days?|months?|hours?))\b|"
    r"\b(?:daily|weekly|monthly|subcutaneously|orally|intravenously)\b",
    re.IGNORECASE,
)

# Matches section numbers (e.g. section 2.1, sec 4, 2.4, 5.1)
_SECTION_NUM_PATTERN = re.compile(
    r"\b(?:section|sec\.?)\s*\d+(?:\.\d+)?\b|\b\d+\.\d+\b",
    re.IGNORECASE,
)


def _get_dynamic_medical_dictionary(user_id: int | None = None) -> set[str]:
    """Build the dynamic medical dictionary merging static terms with indexed drug names."""
    full_dict = set(_BASE_MEDICAL_DICTIONARY)
    for drug in _KNOWN_DRUG_NAMES:
        full_dict.add(drug.lower())
    try:
        indexed_drugs = get_known_drugs(user_id=user_id)
        for drug in indexed_drugs:
            full_dict.add(drug.lower())
    except Exception:
        pass
    return full_dict


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0

    return v0[len(s2)]


def _similarity_ratio(s1: str, s2: str) -> float:
    """Calculate normalized similarity ratio based on edit distance."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = _levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)


def _find_protected_spans(text: str, user_id: int | None = None) -> list[tuple[int, int, str]]:
    """Identify spans in the text that contain protected medical entities."""
    spans: list[tuple[int, int, str]] = []

    # 1. CYP enzymes
    for m in _CYP_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), m.group()))

    # 2. Dosages & units
    for m in _DOSAGE_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), m.group()))

    # 3. Regimens
    for m in _REGIMEN_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), m.group()))

    # 4. Section numbers
    for m in _SECTION_NUM_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), m.group()))

    # 5. Known drug names & acronyms
    all_drugs = set(_KNOWN_DRUG_NAMES)
    try:
        for d in get_known_drugs(user_id=user_id):
            all_drugs.add(d.upper())
    except Exception:
        pass

    for drug in all_drugs:
        pattern = rf"\b{re.escape(drug)}\b"
        for m in re.finditer(pattern, text, re.IGNORECASE):
            spans.append((m.start(), m.end(), m.group()))

    for acr in _PROTECTED_ACRONYMS:
        pattern = rf"\b{re.escape(acr)}\b"
        for m in re.finditer(pattern, text, re.IGNORECASE):
            spans.append((m.start(), m.end(), m.group()))

    # Sort and deduplicate overlapping spans
    spans.sort(key=lambda s: (s[0], -s[1]))
    non_overlapping: list[tuple[int, int, str]] = []
    current_end = -1
    for start, end, val in spans:
        if start >= current_end:
            non_overlapping.append((start, end, val))
            current_end = end

    return non_overlapping


def _is_within_span(start: int, end: int, spans: list[tuple[int, int, str]]) -> bool:
    """Check if token bounds fall inside any protected span."""
    for s_start, s_end, _ in spans:
        if not (end <= s_start or start >= s_end):
            return True
    return False


# Grammatical normalization rules (deterministic agreement / plural / prefix fixes)
_GRAMMAR_REPLACEMENTS = [
    # Question agreement with common variations
    (re.compile(r"\bwhat\s+is\s+(?:contr?a?indicat(?:ions?|ons?)|contrainidcat(?:ions?|ons?)|contra-indicat(?:ions?|ons?))\b", re.I), "what are the contraindications"),
    (re.compile(r"\bwhat\s+is\s+contraindications\b", re.I), "what are the contraindications"),
    (re.compile(r"\bwhat\s+is\s+(?:indicat(?:ions?|ons?)|indcat(?:ions?|ons?))\b", re.I), "what are the indications"),
    (re.compile(r"\bwhat\s+is\s+indications\b", re.I), "what are the indications"),
    (re.compile(r"\bwhat\s+is\s+warnings\b", re.I), "what are the warnings"),
    (re.compile(r"\bwhat\s+is\s+side\s+effects\b", re.I), "what are the side effects"),
    (re.compile(r"\bwhat\s+is\s+adverse\s+reactions\b", re.I), "what are the adverse reactions"),
    (re.compile(r"\bwhat\s+is\s+drug\s+interactions\b", re.I), "what are the drug interactions"),
    (re.compile(r"\bwhat\s+is\s+precautions\b", re.I), "what are the precautions"),
]

# Explicit common medical typo mapping for instant, ultra-high-confidence resolution
_CANONICAL_TYPO_MAP: dict[str, str] = {
    "contrindication": "contraindication",
    "contrindications": "contraindications",
    "contraindicaton": "contraindication",
    "contraindicatons": "contraindications",
    "contra-indication": "contraindication",
    "contra-indications": "contraindications",
    "contraindacation": "contraindication",
    "contraindacations": "contraindications",
    "contrainidcation": "contraindication",
    "contrainidcations": "contraindications",
    "indicaton": "indication",
    "indicatons": "indications",
    "indcation": "indication",
    "indcations": "indications",
    "dosagee": "dosage",
    "dosge": "dosage",
    "adminstration": "administration",
    "administation": "administration",
    "administrtion": "administration",
    "adverese": "adverse",
    "adversee": "adverse",
    "reacton": "reaction",
    "reactons": "reactions",
    "interacton": "interaction",
    "interactons": "interactions",
    "interation": "interaction",
    "interations": "interactions",
    "warnngs": "warnings",
    "warnigns": "warnings",
    "precatuions": "precautions",
    "precautons": "precautions",
    "prescribng": "prescribing",
    "prescrption": "prescription",
    "pheumatoid": "rheumatoid",
    "rheumtoid": "rheumatoid",
    "rheumatod": "rheumatoid",
    "arthrits": "arthritis",
    "arthritiss": "arthritis",
    "psoriatc": "psoriatic",
    "spondylits": "spondylitis",
    "ankylosng": "ankylosing",
    "colits": "colitis",
    "ulceratve": "ulcerative",
    "psoraisis": "psoriasis",
    "suppuratva": "suppurativa",
    "hidradentis": "hidradenitis",
    "uveits": "uveitis",
    "dermititis": "dermatitis",
    "dermatits": "dermatitis",
    "infusionn": "infusion",
    "injecton": "injection",
    "pharmacologcal": "pharmacological",
    "pharmacokinetc": "pharmacokinetic",
    "pharmacodynamc": "pharmacodynamic",
    "tubeculosis": "tuberculosis",
    "tuberculoss": "tuberculosis",
    "thromobsis": "thrombosis",
    "thromobosis": "thrombosis",
    "embolsm": "embolism",
    "hypersensitivty": "hypersensitivity",
    "anaphylaxs": "anaphylaxis",
    "immunosupression": "immunosuppression",
    "perforaton": "perforation",
    "hepatotoxicty": "hepatotoxicity",
    "cardiovasclar": "cardiovascular",
}


def _match_typo_candidate(
    word: str,
    med_dict: set[str],
    protected_tokens: set[str],
) -> tuple[str | None, float, bool]:
    """Find the best single typo correction for a word with conservative matching.

    Returns:
        (best_match, confidence, is_ambiguous)
    """
    word_clean = word.lower().strip()
    if not word_clean or len(word_clean) < 3:
        return None, 1.0, False

    # 1. If in direct protected set or stopwords or already perfectly in dictionary
    if word.upper() in protected_tokens or word_clean in _STOPWORDS_AND_COMMON_WORDS or word_clean in med_dict:
        return None, 1.0, False

    # 2. Check canonical typo map
    if word_clean in _CANONICAL_TYPO_MAP:
        replacement = _CANONICAL_TYPO_MAP[word_clean]
        return replacement, 0.98, False

    # 3. Do not auto-correct very short words (3-4 chars) with edit distance unless explicit
    if len(word_clean) <= 4:
        return None, 1.0, False

    # 4. Search dictionary for closest match with strict criteria
    max_allowed_dist = 1 if len(word_clean) <= 7 else 2
    candidates: list[tuple[str, int, float]] = []

    for term in med_dict:
        # Optimization: length difference cannot exceed max allowed edit distance
        if abs(len(term) - len(word_clean)) > max_allowed_dist:
            continue

        # Prefer candidates that share the first character
        if term[0] != word_clean[0] and len(word_clean) < 8:
            continue

        dist = _levenshtein_distance(word_clean, term)
        if dist <= max_allowed_dist:
            sim = 1.0 - (dist / max(len(word_clean), len(term)))
            min_sim = 0.84 if len(word_clean) <= 7 else 0.80
            if sim >= min_sim:
                candidates.append((term, dist, sim))

    if not candidates:
        return None, 1.0, False

    # Sort by lowest edit distance, then highest similarity
    candidates.sort(key=lambda c: (c[1], -c[2]))

    # Ambiguity / Tie-breaking check:
    # If multiple candidates share the exact same top edit distance and similar score, do NOT guess!
    best_candidate, best_dist, best_sim = candidates[0]
    if len(candidates) > 1:
        second_candidate, second_dist, second_sim = candidates[1]
        if second_dist == best_dist and abs(second_sim - best_sim) < 0.05:
            # Ambiguous! e.g. "huma" could match multiple terms
            return None, 0.40, True

    confidence = round(best_sim, 4)
    return best_candidate, confidence, False


def _check_short_or_ambiguous_query(
    query: str,
    has_conversation_context: bool,
    user_id: int | None = None,
) -> tuple[bool, str | None]:
    """Detect if a query is too incomplete or ambiguous for reliable retrieval."""
    q_trimmed = query.strip().rstrip("?.!").strip()
    words = [w for w in re.split(r"\s+", q_trimmed) if w]

    # If context exists (follow-up in conversation), short queries like "dose?" should pass to context resolver
    if has_conversation_context:
        return False, None

    # Single-word or 2-word queries without context
    if len(words) == 1:
        w = words[0].lower()
        # Single question words or fragments
        if w in {"what", "why", "who", "when", "where", "how", "dose", "dosing", "contra", "contraindication", "warning", "side"}:
            return True, f'Could you please clarify your question about "{q_trimmed}"?'
        # Non-dictionary short mystery words (e.g. "huma")
        known_drugs = {d.lower() for d in get_known_drugs(user_id=user_id)} | {d.lower() for d in _KNOWN_DRUG_NAMES}
        if w not in _BASE_MEDICAL_DICTIONARY and w not in known_drugs and w not in _STOPWORDS_AND_COMMON_WORDS:
            if len(w) <= 5:
                return True, f'Could you clarify what you mean by "{q_trimmed}"?'

    # Queries with pattern "what is <unknown_fragment>" where fragment is ambiguous or incomplete
    m = re.match(r"^what\s+(?:is|are|about)\s+([a-zA-Z0-9_\-]+)\??$", q_trimmed, re.IGNORECASE)
    if m:
        target = m.group(1).lower().strip()
        known_drugs = {d.lower() for d in get_known_drugs(user_id=user_id)} | {d.lower() for d in _KNOWN_DRUG_NAMES}
        if target not in _BASE_MEDICAL_DICTIONARY and target not in known_drugs and target not in _STOPWORDS_AND_COMMON_WORDS:
            if len(target) <= 5:
                return True, f'Could you clarify what you mean by "{m.group(1)}"' + '?'

    return False, None


def normalize_query(
    query: str,
    session_id: str | None = None,
    has_conversation_context: bool = False,
    user_id: int | None = None,
) -> NormalizationResult:
    """Normalize user query with medical preservation, typo correction, and confidence scoring.

    NO LLM call — 100% deterministic and fast.
    """
    if not query or not query.strip():
        return NormalizationResult(
            original_query=query,
            normalized_query="",
            changed=False,
            confidence=1.0,
            corrections=[],
            needs_clarification=False,
        )

    original_query = query
    text = query

    # Step 1: Basic deterministic whitespace and punctuation cleanup
    # Normalize excessive spaces
    text = re.sub(r"[ \t]+", " ", text).strip()
    # Normalize excessive question marks / exclamation marks
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"!{2,}", "!", text)
    # Remove space before punctuation: "rinvoq ?" -> "rinvoq?"
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)

    corrections: list[CorrectionDetail] = []
    overall_confidence = 1.0

    # Step 2: Check for short/ambiguous queries without context
    is_ambiguous, clarif_msg = _check_short_or_ambiguous_query(
        text,
        has_conversation_context=has_conversation_context,
        user_id=user_id,
    )
    if is_ambiguous:
        return NormalizationResult(
            original_query=original_query,
            normalized_query=text,
            changed=False,
            confidence=0.40,
            corrections=[],
            needs_clarification=True,
            clarification_message=clarif_msg,
        )

    # Step 3: Apply safe grammatical agreement transformations
    for pattern, replacement in _GRAMMAR_REPLACEMENTS:
        match = pattern.search(text)
        if match:
            old_segment = match.group()
            text = pattern.sub(replacement, text)
            corrections.append(
                CorrectionDetail(
                    original=old_segment,
                    replacement=replacement,
                    confidence=0.98,
                    reason="grammar_agreement",
                )
            )

    # Step 4: Normalize casing for known drug names
    all_known_drugs = set(_KNOWN_DRUG_NAMES)
    try:
        for d in get_known_drugs(user_id=user_id):
            all_known_drugs.add(d.upper())
    except Exception:
        pass

    for drug in all_known_drugs:
        pattern = rf"\b{re.escape(drug)}\b"
        for m in list(re.finditer(pattern, text, re.IGNORECASE)):
            if m.group() != drug:
                old_val = m.group()
                text = re.sub(rf"\b{re.escape(old_val)}\b", drug, text)
                corrections.append(
                    CorrectionDetail(
                        original=old_val,
                        replacement=drug,
                        confidence=1.0,
                        reason="drug_casing",
                    )
                )

    # Step 5: Identify protected spans (drug names, dosages, CYP enzymes, acronyms, sections)
    protected_spans = _find_protected_spans(text, user_id=user_id)
    protected_entities_found = [span[2] for span in protected_spans]
    protected_tokens = {e.upper() for e in protected_entities_found}

    # Step 6: Dynamic medical dictionary build
    med_dict = _get_dynamic_medical_dictionary(user_id=user_id)

    # Step 7: Word-by-word typo correction outside protected spans
    # Tokenize preserving word positions
    token_iter = re.finditer(r"\b([a-zA-Z0-9_\-\']+)\b", text)
    tokens_to_replace = []

    for match in token_iter:
        word = match.group(1)
        start_idx = match.start(1)
        end_idx = match.end(1)

        # Skip if within protected span
        if _is_within_span(start_idx, end_idx, protected_spans):
            continue

        # Check typo candidate
        rep, conf, ambig = _match_typo_candidate(word, med_dict, protected_tokens)
        if ambig:
            # Word is ambiguous and has multiple interpretations!
            return NormalizationResult(
                original_query=original_query,
                normalized_query=text,
                changed=False,
                confidence=0.42,
                corrections=[],
                needs_clarification=True,
                clarification_message=f'Could you clarify what you mean by "{word}"?',
                protected_entities=protected_entities_found,
            )

        if rep and rep.lower() != word.lower():
            # Preserve capitalization if original word was titlecased / capitalized
            if word.istitle() and not rep.isupper():
                rep = rep.capitalize()
            tokens_to_replace.append((start_idx, end_idx, word, rep, conf, "typo_correction"))

    # Apply word replacements in reverse order of index
    if tokens_to_replace:
        tokens_to_replace.sort(key=lambda t: t[0], reverse=True)
        for s_idx, e_idx, old_w, new_w, conf, reason in tokens_to_replace:
            text = text[:s_idx] + new_w + text[e_idx:]
            corrections.append(
                CorrectionDetail(
                    original=old_w,
                    replacement=new_w,
                    confidence=conf,
                    reason=reason,
                )
            )
            overall_confidence = min(overall_confidence, conf)

    # Step 8: Final whitespace cleanup
    normalized_query = re.sub(r"\s+", " ", text).strip()
    changed = normalized_query != original_query

    # Step 8: Structured Debug Logging
    logger.info(
        "\n--- Query Normalization ---\n"
        "Original Query: %s\n"
        "Normalized Query: %s\n"
        "Changed: %s\n"
        "Confidence: %.2f\n"
        "Corrections: %s\n"
        "Protected Entities: %s\n"
        "---------------------------",
        original_query,
        normalized_query,
        changed,
        overall_confidence,
        [f"{c.original} -> {c.replacement} ({c.confidence})" for c in corrections],
        protected_entities_found,
    )

    return NormalizationResult(
        original_query=original_query,
        normalized_query=normalized_query,
        changed=changed,
        confidence=overall_confidence,
        corrections=corrections,
        needs_clarification=False,
        clarification_message=None,
        protected_entities=protected_entities_found,
    )
