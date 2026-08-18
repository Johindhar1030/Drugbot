"""Regression tests for contextual medicine/coreference resolution.

Validates that "this medicine", "this drug", "it", "its", and multilingual
medicine references are correctly resolved to the active drug from the
current conversation's state — WITHOUT an LLM call (deterministic).
"""

import pytest
from app.rag.context_resolver import (
    _resolve_followup_deterministically,
    ResolutionResult,
)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _state_with_drug(drug: str, last_question: str = None, last_answer: str = None,
                     comparison_entities=None) -> dict:
    """Construct a minimal conversation state dict with the given drug."""
    state = {
        "drug": drug,
        "active_document": None,
        "current_indication": None,
        "current_population": None,
        "current_topic": None,
        "current_section": None,
        "last_answer_entities": [],
        "last_question": last_question,
        "last_answer": last_answer,
    }
    if comparison_entities is not None:
        state["comparison_entities"] = comparison_entities
    return state


def _empty_state() -> dict:
    """Construct a state with no active drug (fresh conversation)."""
    return {
        "drug": None,
        "active_document": None,
        "current_indication": None,
        "current_population": None,
        "current_topic": None,
        "current_section": None,
        "last_answer_entities": [],
        "last_question": None,
        "last_answer": None,
    }


# ════════════════════════════════════════════════════════════════════════════
# Test 1 — Basic "this medicine" reference
# ════════════════════════════════════════════════════════════════════════════

class TestBasicMedicineReference:
    """After discussing SKYRIZI, 'this medicine' should resolve to SKYRIZI."""

    def test_this_medicine_resolves_to_active_drug(self):
        state = _state_with_drug("SKYRIZI", last_question="What is Skyrizi?")
        result = _resolve_followup_deterministically(
            "Can a guy named Manoj use this medicine? His age is 33.",
            state,
        )
        assert result is not None
        assert result.is_followup is True
        assert "SKYRIZI" in result.resolved_query
        assert "this medicine" not in result.resolved_query.lower()

    def test_this_drug_resolves(self):
        state = _state_with_drug("RINVOQ", last_question="What is Rinvoq?")
        result = _resolve_followup_deterministically(
            "What are the side effects of this drug?",
            state,
        )
        assert result is not None
        assert "RINVOQ" in result.resolved_query
        assert "this drug" not in result.resolved_query.lower()

    def test_the_medicine_resolves(self):
        state = _state_with_drug("HUMIRA", last_question="Tell me about Humira")
        result = _resolve_followup_deterministically(
            "Is the medicine safe during pregnancy?",
            state,
        )
        assert result is not None
        assert "HUMIRA" in result.resolved_query

    def test_that_medication_resolves(self):
        state = _state_with_drug("SKYRIZI", last_question="What is Skyrizi?")
        result = _resolve_followup_deterministically(
            "What are the contraindications of that medication?",
            state,
        )
        assert result is not None
        assert "SKYRIZI" in result.resolved_query


# ════════════════════════════════════════════════════════════════════════════
# Test 2 — Possessive pronoun "its"
# ════════════════════════════════════════════════════════════════════════════

class TestPossessivePronoun:
    """'its indications', 'its dosing' should resolve to the active drug."""

    def test_its_indications(self):
        state = _state_with_drug("SKYRIZI", last_question="What is Skyrizi?")
        result = _resolve_followup_deterministically(
            "What are its indications?",
            state,
        )
        assert result is not None
        assert result.is_followup is True
        assert "SKYRIZI" in result.resolved_query
        assert "its indications" not in result.resolved_query.lower()

    def test_its_dosing(self):
        state = _state_with_drug("RINVOQ", last_question="Tell me about Rinvoq")
        result = _resolve_followup_deterministically(
            "What about its dosing?",
            state,
        )
        assert result is not None
        assert "RINVOQ" in result.resolved_query

    def test_its_side_effects(self):
        state = _state_with_drug("HUMIRA", last_question="What is Humira?")
        result = _resolve_followup_deterministically(
            "What are its side effects?",
            state,
        )
        assert result is not None
        assert "HUMIRA" in result.resolved_query


# ════════════════════════════════════════════════════════════════════════════
# Test 3 — Latest medication tracking
# ════════════════════════════════════════════════════════════════════════════

class TestLatestMedicationTracking:
    """After discussing SKYRIZI then RINVOQ, 'its' should resolve to RINVOQ."""

    def test_latest_drug_wins(self):
        # Simulate: discussed SKYRIZI, then switched to RINVOQ
        state = _state_with_drug("RINVOQ", last_question="What is Rinvoq?")
        result = _resolve_followup_deterministically(
            "What are its side effects?",
            state,
        )
        assert result is not None
        assert "RINVOQ" in result.resolved_query
        assert "SKYRIZI" not in result.resolved_query

    def test_this_drug_uses_latest(self):
        state = _state_with_drug("HUMIRA", last_question="Describe Humira")
        result = _resolve_followup_deterministically(
            "Can this drug be used for children?",
            state,
        )
        assert result is not None
        assert "HUMIRA" in result.resolved_query


# ════════════════════════════════════════════════════════════════════════════
# Test 4 — Conversation isolation (no drug → no resolution)
# ════════════════════════════════════════════════════════════════════════════

class TestConversationIsolation:
    """Without a drug in state (fresh conversation), medicine references should NOT resolve."""

    def test_no_drug_no_resolution(self):
        state = _empty_state()
        result = _resolve_followup_deterministically(
            "Can I use this medicine?",
            state,
        )
        # Should return None because there is no drug in state to resolve to
        assert result is None

    def test_no_drug_its_no_resolution(self):
        state = _empty_state()
        result = _resolve_followup_deterministically(
            "What are its indications?",
            state,
        )
        assert result is None


# ════════════════════════════════════════════════════════════════════════════
# Test 5 — Comparison ambiguity
# ════════════════════════════════════════════════════════════════════════════

class TestComparisonAmbiguity:
    """After 'Compare Skyrizi and Rinvoq', 'its dosing' is ambiguous."""

    def test_ambiguous_after_comparison(self):
        state = _state_with_drug(
            "SKYRIZI",
            last_question="Compare Skyrizi and Rinvoq",
            comparison_entities=["plaque psoriasis"],
        )
        result = _resolve_followup_deterministically(
            "What about its dosing?",
            state,
        )
        assert result is not None
        assert result.is_ambiguous is True
        assert result.clarification_question is not None

    def test_explicit_this_medicine_not_ambiguous_after_comparison(self):
        """'this medicine' after a comparison should still resolve (user is explicit)."""
        state = _state_with_drug(
            "SKYRIZI",
            last_question="Compare Skyrizi and Rinvoq",
        )
        result = _resolve_followup_deterministically(
            "Can I use this medicine for psoriasis?",
            state,
        )
        assert result is not None
        # "this medicine" is explicit enough → should resolve, not be ambiguous
        assert result.is_ambiguous is False
        assert "SKYRIZI" in result.resolved_query


# ════════════════════════════════════════════════════════════════════════════
# Test 6 — Multilingual references
# ════════════════════════════════════════════════════════════════════════════

class TestMultilingualReferences:
    """Tamil and Hindi medicine references should resolve to the active drug."""

    def test_tamil_this_medicine(self):
        state = _state_with_drug("SKYRIZI", last_question="ஸ்கைரிசி என்றால் என்ன?")
        result = _resolve_followup_deterministically(
            "இந்த மருந்தை 33 வயது நபர் பயன்படுத்தலாமா?",
            state,
        )
        assert result is not None
        assert "SKYRIZI" in result.resolved_query

    def test_hindi_this_medicine(self):
        state = _state_with_drug("SKYRIZI", last_question="Skyrizi क्या है?")
        result = _resolve_followup_deterministically(
            "क्या यह दवा 33 साल के व्यक्ति के लिए है?",
            state,
        )
        assert result is not None
        assert "SKYRIZI" in result.resolved_query


# ════════════════════════════════════════════════════════════════════════════
# Test 7 — Preserved existing behavior
# ════════════════════════════════════════════════════════════════════════════

class TestExistingBehaviorPreserved:
    """Existing indication-based follow-ups must still work."""

    def test_what_about_indication_still_works(self):
        state = _state_with_drug("HUMIRA", last_question="What is Humira dosage?")
        state["current_topic"] = "dosage and administration"
        result = _resolve_followup_deterministically(
            "What about Crohn's disease?",
            state,
        )
        assert result is not None
        assert "crohn" in result.resolved_query.lower()

    def test_short_topic_still_works(self):
        state = _state_with_drug("HUMIRA", last_question="Tell me about Humira")
        result = _resolve_followup_deterministically(
            "contraindications?",
            state,
        )
        assert result is not None
        assert "HUMIRA" in result.resolved_query
        assert "contraindication" in result.resolved_query.lower()

    def test_standalone_question_returns_none(self):
        """A new standalone question with no references should return None."""
        state = _state_with_drug("SKYRIZI")
        result = _resolve_followup_deterministically(
            "What is the recommended dosage for Rinvoq?",
            state,
        )
        # This is a standalone query mentioning Rinvoq explicitly — not a coreference
        assert result is None
