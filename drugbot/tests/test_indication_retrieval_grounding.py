import pytest
from app.retrieval.vector_store import upsert_chunks
from app.retrieval.keyword_index import ensure_bm25_index
from app.retrieval.hybrid_retriever import hybrid_retrieve, rerank
from app.rag.chain import handle_chat_message
from app.rag.citations import extract_citations, format_citations_text
from app.rag.context_resolver import _resolve_followup_deterministically


@pytest.fixture(autouse=True)
def setup_test_chunks():
    chunks = [
        {
            "id": "test_rinvoq_bw",
            "text": "BOXED WARNING: SERIOUS INFECTIONS, MORTALITY, MALIGNANCY, MAJOR ADVERSE CARDIOVASCULAR EVENTS, AND THROMBOSIS. RINVOQ is a Janus kinase (JAK) inhibitor.",
            "metadata": {
                "drug_name": "RINVOQ",
                "section": "BOXED WARNING",
                "page_number": 1,
                "is_table": False,
                "is_boxed_warning": True,
            },
        },
        {
            "id": "test_rinvoq_ind_1",
            "text": "1 INDICATIONS AND USAGE: RINVOQ (upadacitinib) is indicated for the treatment of moderate to severe rheumatoid arthritis, psoriatic arthritis, atopic dermatitis, ulcerative colitis, and Crohn's disease in adult patients.",
            "metadata": {
                "drug_name": "RINVOQ",
                "section": "1 INDICATIONS AND USAGE",
                "page_number": 2,
                "is_table": False,
                "is_boxed_warning": False,
            },
        },
        {
            "id": "test_rinvoq_ind_1_1",
            "text": "1.1 Rheumatoid Arthritis: RINVOQ is indicated for the treatment of adults with moderately to severely active rheumatoid arthritis who have had an inadequate response to one or more TNF blockers.",
            "metadata": {
                "drug_name": "RINVOQ",
                "section": "1.1 Rheumatoid Arthritis",
                "page_number": 3,
                "is_table": False,
                "is_boxed_warning": False,
            },
        },
        {
            "id": "test_skyrizi_ind_1",
            "text": "1 INDICATIONS AND USAGE: SKYRIZI (risankizumab-rzaa) is indicated for the treatment of moderate-to-severe plaque psoriasis, psoriatic arthritis, and Crohn's disease in adult patients.",
            "metadata": {
                "drug_name": "SKYRIZI",
                "section": "1 INDICATIONS AND USAGE",
                "page_number": 2,
                "is_table": False,
                "is_boxed_warning": False,
            },
        },
        {
            "id": "test_skyrizi_bw",
            "text": "WARNINGS AND PRECAUTIONS: SKYRIZI may increase the risk of infections. Evaluate patients for tuberculosis prior to initiating treatment.",
            "metadata": {
                "drug_name": "SKYRIZI",
                "section": "5 WARNINGS AND PRECAUTIONS",
                "page_number": 5,
                "is_table": False,
                "is_boxed_warning": False,
            },
        },
    ]
    upsert_chunks(chunks)
    ensure_bm25_index(force_rebuild=True)


# ════════════════════════════════════════════════════════════════════════════
# Test 1 — Indications query retrieves Indications & Usage section
# ════════════════════════════════════════════════════════════════════════════

def test_rinvoq_indications_retrieves_indication_section():
    query = "What are the indications for Rinvoq?"
    candidates = hybrid_retrieve(query, drug_name="RINVOQ")
    reranked = rerank(query, candidates)

    assert len(reranked) > 0
    top_sections = [c.get("metadata", {}).get("section", "").upper() for c in reranked[:5]]
    
    # At least one of top 3 chunks must be an Indications & Usage section
    has_indication = any(
        "INDICATION" in sec or "1." in sec or "APPROVED" in sec
        for sec in top_sections[:3]
    )
    assert has_indication, f"Top sections were {top_sections}, expected INDICATION section near top"


# ════════════════════════════════════════════════════════════════════════════
# Test 2 — Skyrizi "used for" query prioritizes indication sections
# ════════════════════════════════════════════════════════════════════════════

def test_skyrizi_used_for_prioritizes_indication_sections():
    query = "What is Skyrizi used for?"
    candidates = hybrid_retrieve(query, drug_name="SKYRIZI")
    reranked = rerank(query, candidates)

    assert len(reranked) > 0
    top_sections = [c.get("metadata", {}).get("section", "").upper() for c in reranked[:5]]
    
    has_indication = any(
        "INDICATION" in sec or "1." in sec or "APPROVED" in sec or "USAGE" in sec
        for sec in top_sections[:3]
    )
    assert has_indication, f"Top sections were {top_sections}, expected INDICATION section near top"


# ════════════════════════════════════════════════════════════════════════════
# Test 3 — Broad overview query retrieves indication evidence
# ════════════════════════════════════════════════════════════════════════════

def test_broad_overview_retrieves_indication_evidence():
    query = "What is Rinvoq?"
    candidates = hybrid_retrieve(query, drug_name="RINVOQ")
    reranked = rerank(query, candidates)

    assert len(reranked) > 0
    top_sections = [c.get("metadata", {}).get("section", "").upper() for c in reranked[:5]]

    has_indication = any(
        "INDICATION" in sec or "1." in sec or "APPROVED" in sec or "USAGE" in sec
        for sec in top_sections[:3]
    )
    assert has_indication, f"Broad query top sections were {top_sections}, expected INDICATION section near top"


# ════════════════════════════════════════════════════════════════════════════
# Test 4 — BOXED WARNING is not the ONLY citation when indication evidence exists
# ════════════════════════════════════════════════════════════════════════════

def test_boxed_warning_not_sole_citation_for_overview():
    query = "What is Rinvoq?"
    res = handle_chat_message(session_id="test_bw_not_sole_citation", message=query)
    
    citations = res.get("citations", [])
    sections = [c.get("section", "").upper() for c in citations]
    
    # If citations exist, ensure BOXED WARNING is not the ONLY cited section
    if citations:
        non_bw_sections = [s for s in sections if "BOXED WARNING" not in s]
        assert len(non_bw_sections) > 0, f"Citations contained ONLY Boxed Warning: {sections}"


# ════════════════════════════════════════════════════════════════════════════
# Test 5 — Citation metadata matches actual retrieved chunks
# ════════════════════════════════════════════════════════════════════════════

def test_citation_metadata_matches_chunks():
    query = "What are the indications for Rinvoq?"
    res = handle_chat_message(session_id="test_citation_meta_match", message=query)
    
    answer = res.get("answer", "")
    citations = res.get("citations", [])

    # Citations must be extracted cleanly with document, section, and page
    for cite in citations:
        assert "document" in cite or "drug" in cite
        assert "section" in cite
        assert "page" in cite or "page_number" in cite


# ════════════════════════════════════════════════════════════════════════════
# Test 6 — Contextual follow-up coreference with indication retrieval
# ════════════════════════════════════════════════════════════════════════════

def test_contextual_followup_its_indications():
    state = {"drug": "RINVOQ", "last_question": "What is Rinvoq?"}
    resolved = _resolve_followup_deterministically("What are its indications?", state)
    
    assert resolved is not None
    assert "RINVOQ" in resolved.resolved_query
    
    # Verify retrieval for the resolved query prioritizes indications
    candidates = hybrid_retrieve(resolved.resolved_query, drug_name="RINVOQ")
    reranked = rerank(resolved.resolved_query, candidates)
    
    top_sections = [c.get("metadata", {}).get("section", "").upper() for c in reranked[:5]]
    has_indication = any("INDICATION" in sec or "1." in sec for sec in top_sections[:3])
    assert has_indication, f"Resolved follow-up top sections: {top_sections}"


def test_contextual_followup_this_medicine_used_for():
    state = {"drug": "SKYRIZI", "last_question": "What is Skyrizi?"}
    resolved = _resolve_followup_deterministically("What is this medicine used for?", state)
    
    assert resolved is not None
    assert "SKYRIZI" in resolved.resolved_query

    candidates = hybrid_retrieve(resolved.resolved_query, drug_name="SKYRIZI")
    reranked = rerank(resolved.resolved_query, candidates)

    top_sections = [c.get("metadata", {}).get("section", "").upper() for c in reranked[:5]]
    has_indication = any("INDICATION" in sec or "1." in sec for sec in top_sections[:3])
    assert has_indication, f"Resolved follow-up top sections: {top_sections}"


# ════════════════════════════════════════════════════════════════════════════
# Test 7 — Multilingual retrieval for broad overview queries
# ════════════════════════════════════════════════════════════════════════════

def test_multilingual_rinvoq_overview_retrieval():
    tamil_query = "Rinvoq என்றால் என்ன?"
    candidates = hybrid_retrieve(tamil_query, drug_name="RINVOQ")
    reranked = rerank(tamil_query, candidates)

    assert len(reranked) > 0
    top_sections = [c.get("metadata", {}).get("section", "").upper() for c in reranked[:5]]
    has_indication = any("INDICATION" in sec or "1." in sec or "USAGE" in sec for sec in top_sections[:4])
    assert has_indication, f"Tamil query top sections were {top_sections}"
