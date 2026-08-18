import json
import logging
import re
from dataclasses import dataclass, field
from langchain_groq import ChatGroq
from app.core.config import settings, groq_llm_kwargs

logger = logging.getLogger(__name__)

@dataclass
class ResolutionResult:
    is_followup: bool
    is_ambiguous: bool
    clarification_question: str | None
    intent: str  # "comparison" | "standard"
    resolved_query: str
    entities: dict = field(default_factory=dict)
    retrieval_queries: list[str] = field(default_factory=list)
    required_sections: list[str] = field(default_factory=list)

# ── Indication Patterns for Decomposition ──────────────────────────────

_INDICATION_PATTERNS = [
    ("rheumatoid arthritis", r"\b(rheumatoid arthritis|ra)\b"),
    ("psoriatic arthritis", r"\b(psoriatic arthritis|psa)\b"),
    ("ankylosing spondylitis", r"\b(ankylosing spondylitis|as)\b"),
    ("Crohn's disease", r"\b(crohn'?s( disease)?|cd)\b"),
    ("ulcerative colitis", r"\b(ulcerative colitis|uc)\b"),
    ("plaque psoriasis", r"\b(plaque psoriasis|psoriasis)\b"),
    ("hidradenitis suppurativa", r"\b(hidradenitis suppurativa|hs)\b"),
    ("uveitis", r"\b(uveitis)\b"),
    ("juvenile idiopathic arthritis", r"\b(juvenile idiopathic arthritis|jia)\b"),
    ("atopic dermatitis", r"\b(atopic dermatitis|ad)\b"),
]


def decompose_multi_indication_query(query: str, drug_name: str | None = None) -> list[str]:
    """Detect multi-indication comparison queries and generate targeted sub-queries.

    NO LLM call — 100% deterministic entity extraction.
    """
    q_lower = query.lower()

    detected = []
    if re.search(r"\b(all indications|8 indications|table of indications|every indication|each indication)\b", q_lower):
        detected = [
            "rheumatoid arthritis", "psoriatic arthritis", "ankylosing spondylitis",
            "Crohn's disease", "ulcerative colitis", "plaque psoriasis",
            "hidradenitis suppurativa", "uveitis"
        ]
    else:
        for canonical_name, pattern in _INDICATION_PATTERNS:
            if re.search(pattern, q_lower, re.IGNORECASE):
                if canonical_name not in detected:
                    detected.append(canonical_name)

    is_multi = bool(re.search(
        r"\b(compare|table|dosing for|dosage for|indications|versus|vs\.?|side by side|which)\b",
        q_lower
    ))

    if (is_multi and len(detected) >= 2) or len(detected) >= 3:
        drug = drug_name or "HUMIRA"
        sub_queries = [query]
        for ind in detected:
            sub_queries.append(f"{drug} recommended dosage {ind}")
        logger.info(
            "Decomposed multi-indication query into %d sub-queries for indications: %s",
            len(sub_queries), detected
        )
        return sub_queries

    return [query]


# ── Compact context resolver prompt ──────────────────────────────────────

CONTEXT_RESOLVER_PROMPT = """\
Analyze this chatbot message given the conversation state. Determine if it's a follow-up, has comparison intent, and rewrite it into a self-contained query.

State: {state_json}
Message: {message}

Rules:
1. is_followup: true if message uses pronouns/references like "it","this","that","what about" or depends on prior context.
2. intent: "comparison" if comparing drugs/indications/populations; else "standard".
3. is_ambiguous: true only if multiple valid interpretations exist and you cannot determine which.
4. resolved_query: rewrite into complete standalone query resolving all pronouns.
5. retrieval_queries: list of 1-2 search strings.
6. entities: {{drug, primary_indication, primary_population, comparison_indication, topic}}.
7. required_sections: likely section numbers (e.g. ["2.4","4"]).

Output ONLY valid JSON:
{{"is_followup":false,"is_ambiguous":false,"clarification_question":null,"intent":"standard","resolved_query":"...","entities":{{}},"retrieval_queries":["..."],"required_sections":[]}}"""


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    return text.strip()

def _parse_resolver_json(raw: str) -> dict | None:
    cleaned = _clean_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None

def _validate_resolver_schema(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    data.setdefault("is_followup", False)
    data.setdefault("is_ambiguous", False)
    data.setdefault("clarification_question", None)
    data.setdefault("intent", "standard")
    data.setdefault("entities", {})
    if not isinstance(data.get("entities"), dict):
        data["entities"] = {}
    if not isinstance(data.get("retrieval_queries"), list):
        data["retrieval_queries"] = [str(data.get("resolved_query", ""))]
    if not isinstance(data.get("required_sections"), list):
        data["required_sections"] = []
    return bool(data.get("resolved_query"))


def _is_simple_standalone_query(message: str) -> bool:
    """Detect if a message is clearly a standalone question (no follow-up resolution needed)."""
    m = message.strip().lower()
    if re.search(r"\b(it|this|that|which|former|latter|what about|how about|compare with)\b", m, re.I):
        return False
    if re.search(r"(what|how|list|describe|compare|tell me)", m, re.I):
        return True
    return False


_MULTI_DRUG_PATTERN = re.compile(
    r"(?:tell me about|describe|what is|what are|information about|info about|about)\s+"
    r"(?:the\s+)?(?:drug(?:s)?\s+(?:called|named)?\s*)?"
    r"([\w\s]+(?:,\s*[\w\s]+){1,})",
    re.IGNORECASE,
)


def extract_multi_drug_query(message: str, user_id: int | None = None) -> list[str]:
    """Extract multiple drug names from a query like 'tell me about actigall, acular, aquadeks'.
    Returns a list of normalized drug names found in the index, or empty list if not a multi-drug query.
    """
    from app.retrieval.vector_store import get_known_drugs, normalize_drug_name
    known = get_known_drugs(user_id=user_id)
    known_lower = {k.lower(): k for k in known}

    # Split on commas and 'and'
    parts = re.split(r"[,]|\band\b", message, flags=re.IGNORECASE)
    found = []
    for part in parts:
        # Strip common lead-in words from the first part
        cleaned = re.sub(
            r"^.*?(?:tell me about|describe|what is|what are|information about|info about|about|drug(?:s)?\s+(?:called|named)?)\s*",
            "", part.strip(), flags=re.IGNORECASE
        ).strip()
        if not cleaned:
            cleaned = part.strip()
        # Try to match against known drugs
        for word in re.split(r"\s+", cleaned):
            w = word.strip().lower().rstrip(".,?!")
            if w in known_lower and known_lower[w] not in found:
                found.append(known_lower[w])
                break
        else:
            # Try the whole cleaned phrase
            phrase = cleaned.lower().strip().rstrip(".,?!")
            if phrase in known_lower and known_lower[phrase] not in found:
                found.append(known_lower[phrase])

    return found if len(found) >= 2 else []


def _detect_comparison_intent(message: str) -> bool:
    """Deterministic comparison detection."""
    return bool(re.search(
        r"\b(compare|versus|vs\.?|difference|how does .+ compare|"
        r"which (has|is|one)|higher|lower|in a table|side.?by.?side)\b",
        message, re.IGNORECASE
    ))


def _resolve_followup_deterministically(message: str, state: dict) -> ResolutionResult | None:
    m = message.strip().lower()
    drug = state.get("drug")
    comparison_entities = state.get("comparison_entities") or []
    if not isinstance(comparison_entities, list):
        comparison_entities = []
    
    if not comparison_entities and state.get("current_indication"):
        comparison_entities = [state["current_indication"]]

    # ── Medicine-Reference Coreference Resolution ──────────────────────────
    # Patterns that reference the previously discussed medicine without naming it.
    # Covers English, Tamil, and Hindi medicine/drug reference phrases.
    _MEDICINE_REF_PATTERN = re.compile(
        r"\b("
        # English
        r"this\s+(?:medicine|medication|drug|treatment)"
        r"|that\s+(?:medicine|medication|drug|treatment)"
        r"|the\s+(?:medicine|medication|drug|treatment)"
        r"|the\s+same\s+(?:medicine|medication|drug|treatment)"
        r"|this\s+med"
        r"|that\s+med"
        # Standalone pronouns — only match when state has a drug
        r"|(?<!\w)it(?!\w)(?!\s+(?:is|was|has|had|can|could|will|would|should|might|may|does|do)\b)"
        r")\b",
        re.IGNORECASE,
    )
    # Possessive pronoun referencing a drug (e.g. "its indications", "its dosing")
    _POSSESSIVE_REF_PATTERN = re.compile(
        r"\bits\s+(?:dosing|dosage|dose|doses|indications?|contraindications?"
        r"|warnings?|precautions?|side\s+effects?|adverse\s+reactions?"
        r"|interactions?|administration|composition|description"
        r"|mechanism|pharmacology|pharmacokinetics?|efficacy|safety"
        r"|approval|use|uses|usage|label|prescribing\s+information)\b",
        re.IGNORECASE,
    )
    # Multilingual medicine references (Tamil, Hindi, and common variants)
    _MULTILINGUAL_MED_REF = re.compile(
        r"("
        # Tamil: இந்த மருந்து / அந்த மருந்து / இந்த மருந்தை / மருந்தின் / மருந்துக்கு / இது
        r"(?:இந்த|அந்த)\s+மருந்[\u0B80-\u0BFF]+"
        r"|இது"
        # Hindi: यह दवा / इस दवा / वह दवा / यह दवाई / इसका / इसकी / इसके
        r"|(?:यह|इस|वह|उस)\s+(?:दवा|दवाई|औषधि)"
        r"|इसक[ाीे]"
        r")",
        re.UNICODE | re.IGNORECASE,
    )

    has_medicine_ref = bool(_MEDICINE_REF_PATTERN.search(m))
    has_possessive_ref = bool(_POSSESSIVE_REF_PATTERN.search(m))
    has_multilingual_ref = bool(_MULTILINGUAL_MED_REF.search(message))  # use original case for Unicode

    if (has_medicine_ref or has_possessive_ref or has_multilingual_ref) and drug:
        # Check for comparison ambiguity: if the last turn was a comparison of 2+ drugs,
        # a single pronoun like "its" is genuinely ambiguous.
        last_q = (state.get("last_question") or "").lower()
        is_prev_comparison = bool(re.search(
            r"\b(compare|versus|vs\.?|difference|and)\b", last_q, re.I
        )) and len(re.findall(
            r"\b(?:SKYRIZI|RINVOQ|HUMIRA|DUPIXENT|STELARA|REMICADE|ENBREL|CIMZIA|COSENTYX|TALTZ|TREMFYA|XELJANZ|OLUMIANT|OTEZLA|ENTYVIO|KEYTRUDA|OPDIVO|SIMPONI|ACTEMRA|KEVZARA|BIMZELX)\b",
            state.get("last_question") or "", re.I
        )) >= 2

        if is_prev_comparison and has_possessive_ref and not has_medicine_ref:
            # Genuinely ambiguous after a comparison — ask for clarification
            return ResolutionResult(
                is_followup=True,
                is_ambiguous=True,
                clarification_question="The previous question compared multiple medications. Could you specify which drug you're asking about?",
                intent="standard",
                resolved_query=message,
                entities={"drug": drug},
                retrieval_queries=[message],
                required_sections=[],
            )

        # Perform substitution: replace medicine references with the actual drug name
        resolved_q = message
        # English substitutions
        resolved_q = re.sub(
            r"\b(?:this|that|the)\s+(?:medicine|medication|drug|treatment|med)\b",
            drug, resolved_q, flags=re.IGNORECASE,
        )
        resolved_q = re.sub(
            r"\bthe\s+same\s+(?:medicine|medication|drug|treatment)\b",
            drug, resolved_q, flags=re.IGNORECASE,
        )
        # Possessive "its" → drug's (or just replace "its X" with "drug X")
        resolved_q = re.sub(
            r"\bits\s+(dosing|dosage|dose|doses|indications?|contraindications?"
            r"|warnings?|precautions?|side\s+effects?|adverse\s+reactions?"
            r"|interactions?|administration|composition|description"
            r"|mechanism|pharmacology|pharmacokinetics?|efficacy|safety"
            r"|approval|use|uses|usage|label|prescribing\s+information)\b",
            rf"{drug} \1", resolved_q, flags=re.IGNORECASE,
        )
        # Standalone "it" → drug name (careful: only when not followed by is/was/etc
        # which would form a valid independent clause)
        resolved_q = re.sub(
            r"\bit\b(?!\s+(?:is|was|has|had|can|could|will|would|should|might|may|does|do)\b)",
            drug, resolved_q, flags=re.IGNORECASE,
        )
        # Tamil substitutions
        resolved_q = re.sub(r"(?:இந்த|அந்த)\s+மருந்[\u0B80-\u0BFF]+", drug, resolved_q)
        resolved_q = re.sub(r"இது", drug, resolved_q)
        # Hindi substitutions
        resolved_q = re.sub(r"(?:यह|इस|वह|उस)\s+(?:दवा|दवाई|औषधि)", drug, resolved_q)
        resolved_q = re.sub(r"इसक[ाीे]", f"{drug} का", resolved_q)

        resolved_q = re.sub(r"\s+", " ", resolved_q).strip()

        logger.info(
            "Medicine-reference coreference resolved: '%s' → '%s' (drug=%s)",
            message, resolved_q, drug,
        )

        return ResolutionResult(
            is_followup=True,
            is_ambiguous=False,
            clarification_question=None,
            intent="standard",
            resolved_query=resolved_q,
            entities={"drug": drug},
            retrieval_queries=[resolved_q],
            required_sections=[],
        )

    # ── Existing Indication / Comparison Resolution ────────────────────────
    # Default drug for indication-based resolution (fallback only)
    if not drug:
        drug = "HUMIRA"

    is_anaphora_which = bool(re.search(r"\b(which (one|of these|indication|has|is)|the former|the latter)\b", m, re.I))
    is_compare_phrase = bool(re.search(r"\b(how does that compare|that compare|compare with|versus|vs\.?)\b", m, re.I))
    is_what_about = bool(re.search(r"\b(what about|how about)\b", m, re.I))

    new_inds = []
    for canonical_name, pattern in _INDICATION_PATTERNS:
        if re.search(pattern, m, re.I) and canonical_name not in new_inds:
            new_inds.append(canonical_name)

    all_entities = list(comparison_entities)
    for ind in new_inds:
        if ind not in all_entities:
            all_entities.append(ind)

    if is_anaphora_which and all_entities:
        topic_phrase = "initial dose" if re.search(r"initial", m, re.I) else "recommended dosage"
        entities_str = ", ".join(all_entities)
        resolved_q = f"Which of {entities_str} has the higher {topic_phrase} for {drug}? Compare initial and recommended doses."
        sub_qs = [resolved_q] + [f"{drug} recommended dosage {e}" for e in all_entities]
        return ResolutionResult(
            is_followup=True,
            is_ambiguous=False,
            clarification_question=None,
            intent="comparison",
            resolved_query=resolved_q,
            entities={"drug": drug, "comparison_entities": all_entities, "topic": topic_phrase},
            retrieval_queries=sub_qs,
            required_sections=["2.2", "2.4", "2.5", "2.6", "2.7"]
        )

    if is_compare_phrase and all_entities:
        entities_str = ", ".join(all_entities)
        resolved_q = f"Compare recommended {drug} dosage for {entities_str}."
        sub_qs = [resolved_q] + [f"{drug} recommended dosage {e}" for e in all_entities]
        return ResolutionResult(
            is_followup=True,
            is_ambiguous=False,
            clarification_question=None,
            intent="comparison",
            resolved_query=resolved_q,
            entities={"drug": drug, "comparison_entities": all_entities, "topic": "dosage"},
            retrieval_queries=sub_qs,
            required_sections=["2.2", "2.4", "2.5", "2.6", "2.7"]
        )

    if is_what_about and new_inds and state.get("current_topic"):
        topic = state.get("current_topic") or "dosage and administration"
        ind = new_inds[0]
        resolved_q = f"{drug} {topic} for {ind}"
        sub_qs = [resolved_q]
        if comparison_entities:
            all_entities = list(comparison_entities)
            if ind not in all_entities:
                all_entities.append(ind)
            sub_qs.append(f"Compare {drug} dosage for {', '.join(all_entities)}")
        return ResolutionResult(
            is_followup=True,
            is_ambiguous=False,
            clarification_question=None,
            intent="standard" if len(all_entities) < 2 else "comparison",
            resolved_query=resolved_q,
            entities={"drug": drug, "primary_indication": ind, "comparison_entities": all_entities, "topic": topic},
            retrieval_queries=sub_qs,
            required_sections=[]
        )

    # Check if short topic follow-up like "dose?", "what dose?", "dosage?", "contraindications?", "warnings?"
    is_short_topic = bool(re.search(r"^(?:what\s+(?:is\s+the\s+)?)?(?:dose|dosage|dosing|contraindications?|warnings?|side\s+effects?|adverse\s+reactions?|interactions?|indications?)\??$", m, re.I))
    if is_short_topic and (state.get("last_question") or state.get("drug")):
        topic_detected = "dosage"
        if re.search(r"contraindicat", m, re.I):
            topic_detected = "contraindications"
        elif re.search(r"warning", m, re.I):
            topic_detected = "warnings and precautions"
        elif re.search(r"side\s+effect|adverse", m, re.I):
            topic_detected = "adverse reactions"
        elif re.search(r"interaction", m, re.I):
            topic_detected = "drug interactions"
        elif re.search(r"indication", m, re.I):
            topic_detected = "indications and usage"

        ind = state.get("current_indication")
        if ind:
            resolved_q = f"What is the {drug} {topic_detected} for {ind}?"
        else:
            resolved_q = f"What is the {drug} {topic_detected}?"

        return ResolutionResult(
            is_followup=True,
            is_ambiguous=False,
            clarification_question=None,
            intent="standard",
            resolved_query=resolved_q,
            entities={"drug": drug, "primary_indication": ind, "topic": topic_detected} if ind else {"drug": drug, "topic": topic_detected},
            retrieval_queries=[resolved_q],
            required_sections=[]
        )

    return None


def resolve_context(message: str, state: dict, llm: ChatGroq, user_id: int | None = None) -> ResolutionResult:
    # 0. Check for multi-drug query first (e.g. "tell me about actigall, acular, aquadeks")
    multi_drugs = extract_multi_drug_query(message, user_id=user_id)
    if multi_drugs:
        resolved_q = f"Provide information about each of the following drugs: {', '.join(multi_drugs)}."
        retrieval_qs = [resolved_q] + [f"overview indications usage {d}" for d in multi_drugs]
        logger.info("Multi-drug query detected: %s", multi_drugs)
        return ResolutionResult(
            is_followup=False,
            is_ambiguous=False,
            clarification_question=None,
            intent="multi_drug",
            resolved_query=resolved_q,
            entities={"drugs": multi_drugs},
            retrieval_queries=retrieval_qs,
            required_sections=[],
        )

    # 1. Check deterministic anaphora / follow-up first
    det_res = _resolve_followup_deterministically(message, state)
    if det_res:
        logger.info("Context resolver resolved follow-up deterministically: resolved_query='%s'", det_res.resolved_query)
        return det_res

    # 2. For simple standalone queries with no state, skip LLM call
    if _is_simple_standalone_query(message) and not state.get("last_question"):
        intent = "comparison" if _detect_comparison_intent(message) else "standard"
        return ResolutionResult(
            is_followup=False,
            is_ambiguous=False,
            clarification_question=None,
            intent=intent,
            resolved_query=message,
            entities={},
            retrieval_queries=[message] if intent == "standard" else [message],
            required_sections=[]
        )

    # Compact state: only include non-null fields
    compact_state = {k: v for k, v in state.items() if v is not None and v != [] and k != "last_answer"}
    state_json = json.dumps(compact_state, indent=1) if compact_state else "{}"
    prompt = CONTEXT_RESOLVER_PROMPT.format(state_json=state_json, message=message)

    try:
        from app.core.llm_retry import retry_llm_call
        response = retry_llm_call(llm.invoke, prompt, label="context_resolver")
        data = _parse_resolver_json(response.content)
        if data and _validate_resolver_schema(data):
            return ResolutionResult(
                is_followup=bool(data["is_followup"]),
                is_ambiguous=bool(data["is_ambiguous"]),
                clarification_question=data.get("clarification_question"),
                intent=data["intent"],
                resolved_query=data["resolved_query"],
                entities=data["entities"],
                retrieval_queries=data["retrieval_queries"],
                required_sections=data["required_sections"]
            )
        logger.warning("Resolver: Invalid JSON or schema from LLM")
    except Exception as e:
        logger.warning("Resolver LLM failed: %s", e)

    # Deterministic fallback
    logger.warning("Context resolver falling back to deterministic rules")
    intent = "comparison" if _detect_comparison_intent(message) else "standard"
    return ResolutionResult(
        is_followup=False,
        is_ambiguous=False,
        clarification_question=None,
        intent=intent,
        resolved_query=message,
        entities={},
        retrieval_queries=[message],
        required_sections=[]
    )


def update_conversation_state(
    session_id: str,
    message: str,
    resolved_query: str,
    answer: str,
    chunks: list[dict],
    llm: ChatGroq
) -> dict:
    """Extract conversation state deterministically — NO LLM call.

    Saves ~1500-3000 tokens per request by parsing state from chunk metadata
    and the answer text directly, instead of calling the LLM.
    """
    from app.rag.memory import memory
    prev_state = memory.get_state(session_id)

    # Extract drug from chunks
    drug = prev_state.get("drug")
    sections = []
    for c in chunks:
        meta = c.get("metadata") or {}
        if meta.get("drug_name") and not drug:
            drug = meta["drug_name"]
        sec = meta.get("section")
        if sec and sec not in sections:
            sections.append(sec)

    # Detect topic from the query
    q = (resolved_query + " " + message).lower()
    topic = prev_state.get("current_topic")
    topic_map = [
        (r"dosag|dosing|dose|administration|initial dose", "dosage and administration"),
        (r"contraindicat", "contraindications"),
        (r"warning|precaution", "warnings and precautions"),
        (r"adverse|side effect", "adverse reactions"),
        (r"indication|approved|used for", "indications and usage"),
        (r"description|active ingredient|composition", "description"),
        (r"interaction", "drug interactions"),
        (r"mechanism|pharmacol", "clinical pharmacology"),
    ]
    for pattern, t in topic_map:
        if re.search(pattern, q, re.I):
            topic = t
            break

    # Extract ALL indications in resolved_query, message, or answer
    detected_inds = []
    full_text = (resolved_query + " " + message + " " + answer).lower()
    for canonical_name, pattern in _INDICATION_PATTERNS:
        if re.search(pattern, full_text, re.I):
            if canonical_name not in detected_inds:
                detected_inds.append(canonical_name)

    prev_comparison = prev_state.get("comparison_entities") or []
    if not isinstance(prev_comparison, list):
        prev_comparison = []

    new_comparison = list(prev_comparison)
    for ind in detected_inds:
        if ind not in new_comparison:
            new_comparison.append(ind)

    updates = {
        "last_question": message,
        "last_answer": answer[:200],  # Truncate to save memory
    }
    if drug:
        updates["drug"] = drug
    if topic:
        updates["current_topic"] = topic
    if detected_inds:
        updates["current_indication"] = detected_inds[-1]
    if new_comparison:
        updates["comparison_entities"] = new_comparison
    if sections:
        updates["current_section"] = sections[0]

    memory.update_state(session_id, updates)
    return memory.get_state(session_id)
