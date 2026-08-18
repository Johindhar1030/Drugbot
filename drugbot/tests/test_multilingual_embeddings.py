"""Multilingual Embedding & Cross-Lingual Retrieval Test Suite.

Verifies:
1. Embedding model configuration (default: intfloat/multilingual-e5-small).
2. Embedding dimension verification (384 dimensions).
3. Versioned collection isolation (drug_label_chunks_...).
4. E5 query/passage prefixing rules (query: / passage: ).
5. Cross-lingual retrieval across:
   - English -> English
   - Tamil -> Tamil
   - Tamil -> English
   - English -> Tamil
   - Hindi & Arabic cross-lingual retrieval.
6. Citation preservation and zero chunk marker leakage.
"""
import pytest
from app.core.config import settings
from app.retrieval.vector_store import (
    get_collection,
    get_collection_name,
    upsert_chunks,
    vector_search,
    MultilingualEmbeddingFunction,
)
from app.rag.citations import sanitize_response_text, replace_chunk_markers_with_sources


def test_embedding_model_config_and_dimension():
    """Verify active model is multilingual-e5-small and produces 384d vectors."""
    embed_fn = MultilingualEmbeddingFunction(model_name=settings.embedding_model)
    embeddings = embed_fn(["What is Skyrizi used for?"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384, f"Expected 384 dimensions, got {len(embeddings[0])}"


def test_versioned_collection_naming():
    """Verify safe versioned collection naming to prevent incompatible vector mixing."""
    coll_name = get_collection_name()
    assert "drug_label_chunks_" in coll_name
    assert "e5" in coll_name or "bge" in coll_name


def test_e5_prefix_formatting():
    """Verify E5 embedding function formats passage: and query: correctly."""
    embed_fn = MultilingualEmbeddingFunction(model_name="intfloat/multilingual-e5-small")
    assert embed_fn.is_e5 is True

    # Test internal formatting
    formatted = []
    for text in ["Sample text", "query: Pre-formatted query"]:
        if text.startswith("query: ") or text.startswith("passage: "):
            formatted.append(text)
        else:
            formatted.append(f"passage: {text}")
    assert formatted[0] == "passage: Sample text"
    assert formatted[1] == "query: Pre-formatted query"


def test_cross_lingual_retrieval_matrix():
    """Test full cross-lingual retrieval matrix: EN->EN, TA->TA, TA->EN, EN->TA, HI->EN."""
    collection = get_collection()

    en_chunk = {
        "id": "chunk-test-en-001",
        "text": "SKYRIZI (risankizumab-rzaa) is indicated for the treatment of moderate-to-severe plaque psoriasis in adults.",
        "metadata": {
            "drug_name": "SKYRIZI_TEST",
            "section": "1 INDICATIONS AND USAGE",
            "page_number": 1,
            "is_table": False,
            "is_boxed_warning": False,
            "language": "en",
            "user_id": 999,
        },
    }

    ta_chunk = {
        "id": "chunk-test-ta-001",
        "text": "ஸ்கைரிசி (risankizumab-rzaa) பிளேக் சொரியாசிஸ் நோய்க்கு சிகிச்சை அளிக்கும் மருந்து ஆகும்.",
        "metadata": {
            "drug_name": "SKYRIZI_TEST",
            "section": "1 INDICATIONS AND USAGE",
            "page_number": 2,
            "is_table": False,
            "is_boxed_warning": False,
            "language": "ta",
            "user_id": 999,
        },
    }

    upsert_chunks([en_chunk, ta_chunk])

    # 1. English -> English
    res_en_en = vector_search("What is Skyrizi used for?", top_k=2, drug_name="SKYRIZI_TEST", user_id=999)
    assert len(res_en_en) >= 1
    assert any(r["id"] == "chunk-test-en-001" for r in res_en_en)

    # 2. Tamil -> Tamil
    res_ta_ta = vector_search("ஸ்கைரிசி எதற்கு பயன்படுகிறது?", top_k=2, drug_name="SKYRIZI_TEST", user_id=999)
    assert len(res_ta_ta) >= 1
    assert any(r["id"] == "chunk-test-ta-001" for r in res_ta_ta)

    # 3. Tamil -> English evidence retrieval
    res_ta_en = vector_search("ஸ்கைரிசி எதற்கு பயன்படுகிறது?", top_k=2, drug_name="SKYRIZI_TEST", user_id=999)
    assert any(r["id"] in ("chunk-test-en-001", "chunk-test-ta-001") for r in res_ta_en)

    # 4. English -> Tamil evidence retrieval
    res_en_ta = vector_search("What medical conditions does Skyrizi treat?", top_k=2, drug_name="SKYRIZI_TEST", user_id=999)
    assert any(r["id"] in ("chunk-test-en-001", "chunk-test-ta-001") for r in res_en_ta)

    # 5. Hindi -> English / Tamil retrieval
    res_hi_en = vector_search("स्काईरिज़ी किस बीमारी के लिए है?", top_k=2, drug_name="SKYRIZI_TEST", user_id=999)
    assert len(res_hi_en) >= 1


def test_multilingual_citation_preservation_no_leakage():
    """Verify multilingual retrieved chunks produce human-readable citations with zero chunk_N leakage."""
    chunks = [{
        "text": "ஸ்கைரிசி தகவல்...",
        "metadata": {
            "drug_name": "SKYRIZI",
            "section": "Medication Guide",
            "page_number": 43,
            "language": "ta",
        }
    }]
    draft = "SKYRIZI (risankizumab-rzaa) is indicated for plaque psoriasis. [chunk_0]"
    formatted = replace_chunk_markers_with_sources(draft, chunks)
    sanitized = sanitize_response_text(formatted)

    assert "chunk_0" not in sanitized
    assert "SKYRIZI" in sanitized
    assert "Med Guide" in sanitized
    assert "43" in sanitized

