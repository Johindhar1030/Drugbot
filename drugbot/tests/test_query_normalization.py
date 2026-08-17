"""Unit and integration tests for Controlled Query Normalization & Typo Handling."""
import pytest
import uuid

from app.rag.normalizer import (
    normalize_query,
    NormalizationResult,
    _levenshtein_distance,
    _similarity_ratio,
)
from app.rag.chain import handle_chat_message
from app.rag.memory import memory


class TestQueryNormalizationUnit:
    """Unit tests targeting individual normalization components."""

    def test_1_exact_query_no_unnecessary_normalization(self):
        """Test 1 — Exact query: What are the contraindications of RINVOQ?"""
        q = "What are the contraindications of RINVOQ?"
        res = normalize_query(q)
        assert res.normalized_query == "What are the contraindications of RINVOQ?"
        assert res.changed is False
        assert res.confidence == 1.0
        assert res.needs_clarification is False
        assert "RINVOQ" in res.protected_entities

    def test_2_simple_typo_contraindications(self):
        """Test 2 — Simple typo: What are the contrindications of RINVOQ?"""
        q = "What are the contrindications of RINVOQ?"
        res = normalize_query(q)
        assert res.normalized_query == "What are the contraindications of RINVOQ?"
        assert res.changed is True
        assert res.confidence >= 0.90
        assert res.needs_clarification is False
        assert any(c.original.lower() == "contrindications" and c.replacement.lower() == "contraindications" for c in res.corrections)

    def test_3_another_typo_indications(self):
        """Test 3 — Another typo: What are the indicatons for RINVOQ?"""
        q = "What are the indicatons for RINVOQ?"
        res = normalize_query(q)
        assert res.normalized_query == "What are the indications for RINVOQ?"
        assert res.changed is True
        assert res.confidence >= 0.90
        assert res.needs_clarification is False
        assert any(c.original.lower() == "indicatons" and c.replacement.lower() == "indications" for c in res.corrections)

    def test_4_whitespace_and_punctuation_cleanup(self):
        """Test 4 — Whitespace: What   is   RINVOQ?"""
        q = "What   is   RINVOQ  ??"
        res = normalize_query(q)
        assert res.normalized_query == "What is RINVOQ?"
        assert res.changed is True
        assert res.needs_clarification is False

    def test_5_drug_name_preservation(self):
        """Test 5 — Drug name preservation: What is the dose of RINVOQ?"""
        q = "What is the dose of RINVOQ?"
        res = normalize_query(q)
        assert res.normalized_query == "What is the dose of RINVOQ?"
        assert "RINVOQ" in res.protected_entities
        assert res.changed is False
        assert res.needs_clarification is False

    def test_6_medical_abbreviation_and_enzyme_preservation(self):
        """Test 6 — Medical abbreviation & enzyme preservation: What is the CYP3A4 interaction with RINVOQ?"""
        q = "What is the CYP3A4 interaction with RINVOQ?"
        res = normalize_query(q)
        assert res.normalized_query == "What is the CYP3A4 interaction with RINVOQ?"
        assert "CYP3A4" in res.protected_entities
        assert "RINVOQ" in res.protected_entities
        assert res.changed is False
        assert res.needs_clarification is False

    def test_7_dosage_and_regimen_preservation(self):
        """Test 7 — Dosage preservation: Is 30 mg once daily recommended?"""
        q = "Is 30 mg once daily recommended?"
        res = normalize_query(q)
        assert "30 mg" in res.normalized_query
        assert "once daily" in res.normalized_query
        assert res.changed is False
        assert res.needs_clarification is False

    def test_8_ambiguous_query_clarification(self):
        """Test 8 — Ambiguous query: what is huma"""
        q = "what is huma"
        res = normalize_query(q, has_conversation_context=False)
        assert res.needs_clarification is True
        assert res.clarification_message is not None
        assert "huma" in res.clarification_message.lower()
        # Must NOT auto-convert huma to humira or humari
        assert "humari" not in res.normalized_query.lower()
        assert "humira" not in res.normalized_query.lower()

    def test_9_contextual_follow_up_short_query(self):
        """Test 9 — Contextual follow-up: 'dose?' should not be blocked when conversation context exists."""
        q = "dose?"
        res = normalize_query(q, has_conversation_context=True)
        # Should not force clarification when context is present
        assert res.needs_clarification is False

    def test_10_no_modification_of_medical_intent(self):
        """Test 10 — Normalization must never modify medical facts like dosages or numbers."""
        q = "Is 15 mg once daily or 45 mg once daily used for Crohn's disease?"
        res = normalize_query(q)
        assert "15 mg once daily" in res.normalized_query
        assert "45 mg once daily" in res.normalized_query
        assert "Crohn's disease" in res.normalized_query or "crohn's disease" in res.normalized_query.lower()

    def test_11_grammar_agreement_normalization(self):
        """Test grammar agreement normalization: 'what is contrindications of rinvoq?'"""
        q = "what is contrindications of rinvoq?"
        res = normalize_query(q)
        assert res.normalized_query == "what are the contraindications of RINVOQ?"
        assert res.changed is True
        assert res.needs_clarification is False

    def test_12_multiple_typos_in_one_query(self):
        """Test multiple typos in single query: 'what is the dosge and adminstration for rinvoq?'"""
        q = "what is the dosge and adminstration for rinvoq?"
        res = normalize_query(q)
        assert res.normalized_query == "what is the dosage and administration for RINVOQ?"
        assert res.changed is True
        assert res.confidence >= 0.90
        assert len(res.corrections) >= 2

    def test_13_levenshtein_and_similarity_helper(self):
        """Test edit distance helper functions."""
        assert _levenshtein_distance("rinvoq", "rinvoq") == 0
        assert _levenshtein_distance("indicatons", "indications") == 1
        assert _similarity_ratio("indicatons", "indications") > 0.90


class TestQueryNormalizationIntegration:
    """End-to-end integration tests using handle_chat_message."""

    def test_pipeline_handles_simple_typo(self):
        """Typo query is normalized and successfully retrieved."""
        session_id = str(uuid.uuid4())
        q = "What are the contrindications of RINVOQ?"
        res = handle_chat_message(session_id, q, drug_name_hint="RINVOQ")
        assert res["original_query"] == q
        assert "contraindications" in res["normalized_query"]
        assert res["confidence"] in ["grounded", "not_found", "limited_evidence"]
        assert res["normalization_info"] is not None
        assert res["normalization_info"]["changed"] is True

    def test_pipeline_ambiguous_short_query_returns_clarification(self):
        """Ambiguous query with no prior context returns clarification, not 'Not Found'."""
        session_id = str(uuid.uuid4())
        q = "what is huma"
        res = handle_chat_message(session_id, q)
        assert res["confidence"] == "clarification"
        assert "huma" in res["answer"].lower()
        # Must not say 'couldn't find sufficient evidence' or 'Not Found'
        assert "couldn't find" not in res["answer"].lower()

    def test_pipeline_followup_short_query_with_context(self):
        """Follow-up 'dose?' after dosage context is properly resolved."""
        session_id = str(uuid.uuid4())
        # Turn 1: Set context
        q1 = "What is the RINVOQ dosage for rheumatoid arthritis?"
        res1 = handle_chat_message(session_id, q1, drug_name_hint="RINVOQ")
        assert res1["confidence"] in ["grounded", "not_found", "limited_evidence"]

        # Turn 2: Short follow-up
        q2 = "dose?"
        res2 = handle_chat_message(session_id, q2)
        # Should NOT return an ambiguous rejection error
        assert res2["confidence"] != "ambiguous_query"
