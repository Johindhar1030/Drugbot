"""Regression test suite for internal chunk citation leakage prevention.

Ensures:
1. Internal chunk identifiers like `chunk_0`, `chunk_1`, `chunk_25`, `【chunk_0】`, `[chunk_0]`, `(chunk_0)`
   are NEVER exposed in user-facing chatbot text.
2. Internal `chunk_ref` (e.g. "chunk_0") is preserved in the citations metadata array for internal tracking/debugging.
3. Clean document titles (e.g. "skyrizi_pi.pdf" -> "SKYRIZI Prescribing Information") and human-readable citations are formatted correctly.
4. Response sanitization filter removes lingering internal chunk references without damaging valid medical text.
"""
import re
import pytest
from app.rag.citations import (
    clean_document_name,
    sanitize_response_text,
    replace_chunk_markers_with_sources,
    strip_citation_markers,
    extract_citations,
    format_structured_citation_block,
)


def test_clean_document_name():
    assert clean_document_name("skyrizi_pi.pdf") == "SKYRIZI Prescribing Information"
    assert clean_document_name("rinvoq_pi.pdf") == "RINVOQ Prescribing Information"
    assert clean_document_name("skyrizi") == "SKYRIZI Prescribing Information"
    assert clean_document_name("SKYRIZI Prescribing Information") == "SKYRIZI Prescribing Information"
    assert clean_document_name("humira_label.pdf") == "HUMIRA Prescribing Information"
    assert clean_document_name(None) == "Prescribing Information"


def test_sanitize_response_text_removes_chunk_patterns():
    dirty_text = (
        "SKYRIZI® (risankizumab-rzaa) is a prescription medicine...\n"
        "【chunk_0】\n\n"
        "[chunk_1] [chunk_25] (chunk_123) chunk_99\n"
        "§MEDICATION GUIDE p. 43\n"
        "Document Evidence: High"
    )

    clean_text = sanitize_response_text(dirty_text)

    # Must NOT contain any internal chunk patterns
    assert not re.search(r"chunk_\d+", clean_text, re.IGNORECASE)
    assert not re.search(r"【chunk_\d+】", clean_text)
    assert not re.search(r"\[chunk_\d+\]", clean_text)

    # Must PRESERVE valid content
    assert "SKYRIZI® (risankizumab-rzaa)" in clean_text
    assert "MEDICATION GUIDE" in clean_text
    assert "p. 43" in clean_text
    assert "Document Evidence: High" in clean_text


def test_replace_chunk_markers_with_sources():
    chunks = [
        {
            "metadata": {
                "drug_name": "skyrizi_pi.pdf",
                "section": "MEDICATION GUIDE",
                "page_number": 43,
            }
        },
        {
            "metadata": {
                "drug_name": "rinvoq",
                "section": "2.2 Recommended Dosage",
                "page_number": 5,
            }
        },
    ]

    draft = "Skyrizi is indicated for Crohn's Disease [chunk_0]. Rinvoq is indicated for RA 【chunk_1】."
    formatted = replace_chunk_markers_with_sources(draft, chunks)

    # Assert raw chunk_N markers are replaced by human-readable citations
    assert "chunk_0" not in formatted
    assert "chunk_1" not in formatted
    assert "Med Guide" in formatted or "MEDICATION GUIDE" in formatted
    assert "p.43" in formatted or "p. 43" in formatted
    assert "§2.2" in formatted


def test_internal_chunk_ref_preserved_in_metadata():
    chunks = [
        {
            "metadata": {
                "drug_name": "skyrizi_pi.pdf",
                "section": "MEDICATION GUIDE",
                "page_number": 43,
            }
        }
    ]

    draft = "SKYRIZI® is indicated for plaque psoriasis [chunk_0]."
    citations = extract_citations(draft, chunks)

    assert len(citations) == 1
    # Internal chunk reference preserved for debugging/tracing
    assert citations[0]["chunk_ref"] == "chunk_0"
    # User-facing fields cleanly formatted
    assert citations[0]["document"] == "SKYRIZI Prescribing Information"
    assert citations[0]["section"] == "MEDICATION GUIDE"
    assert citations[0]["page"] == 43


def test_format_structured_citation_block():
    citations = [
        {
            "document": "SKYRIZI Prescribing Information",
            "section": "Medication Guide",
            "page": 43,
        }
    ]
    block = format_structured_citation_block(citations)
    assert "**Source:** SKYRIZI Prescribing Information" in block
    assert "**Section:** Medication Guide" in block
    assert "**Page:** 43" in block


def test_multiple_chunks_leakage_assertion():
    chunks = [
        {"metadata": {"drug_name": "skyrizi", "section": "Medication Guide", "page_number": 43}},
        {"metadata": {"drug_name": "skyrizi", "section": "1 INDICATIONS AND USAGE", "page_number": 1}},
        {"metadata": {"drug_name": "rinvoq", "section": "2.2 Recommended Dosage", "page_number": 5}},
        {"metadata": {"drug_name": "rinvoq", "section": "4 CONTRAINDICATIONS", "page_number": 10}},
    ]
    raw_draft = (
        "Skyrizi treat plaque psoriasis [chunk_0]. It is also for CD [chunk_1].\n"
        "Rinvoq dosage is 15mg [chunk_2]. Contraindications include [chunk_3]. 【chunk_0】"
    )

    formatted = replace_chunk_markers_with_sources(raw_draft, chunks)
    sanitized = sanitize_response_text(formatted)

    # General pattern assertions required by prompt item 11
    assert not re.search(r"chunk_\d+", sanitized, re.IGNORECASE)
    assert not re.search(r"【chunk_\d+】", sanitized)
    assert "chunk_0" not in sanitized
    assert "chunk_1" not in sanitized
    assert "chunk_2" not in sanitized
    assert "chunk_3" not in sanitized

    # Valid citation metadata remains intact
    citations = extract_citations(raw_draft, chunks)
    assert len(citations) == 4
    assert [c["chunk_ref"] for c in citations] == ["chunk_0", "chunk_1", "chunk_2", "chunk_3"]
    assert citations[0]["document"] == "SKYRIZI Prescribing Information"
    assert citations[2]["document"] == "RINVOQ Prescribing Information"


def test_citation_formatting_and_deduplication_regression():
    chunks = [
        {"metadata": {"drug_name": "skyrizi_pi.pdf", "section": "Medication Guide", "page_number": 43}},
        {"metadata": {"drug_name": "skyrizi_pi.pdf", "section": "Medication Guide", "page_number": 43}},
        {"metadata": {"drug_name": "skyrizi_pi.pdf", "section": "1.1 Plaque Psoriasis", "page_number": 3}},
    ]
    raw_draft = "Skyrizi details [chunk_0]. More info [chunk_1]. Psoriasis info [chunk_2]."

    # 1. Test stripping chunk markers creates clean answer text without concatenation artifacts or inline tags
    clean_text = strip_citation_markers(raw_draft)
    assert "§MEDICATION GUIDE p. 43§1.1" not in clean_text
    assert "[Med Guide, p.43]" not in clean_text
    assert "chunk_0" not in clean_text

    # 2. Test extract_citations deduplicates (chunk_0 and chunk_1 point to same doc+sec+page)
    citations = extract_citations(raw_draft, chunks)
    assert len(citations) == 2  # Deduplicated from 3 to 2
    assert citations[0]["section"] == "Medication Guide"
    assert citations[0]["page"] == 43
    assert citations[1]["section"] == "1.1 Plaque Psoriasis"
    assert citations[1]["page"] == 3


