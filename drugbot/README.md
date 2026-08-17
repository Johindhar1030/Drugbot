# Drug Information Chatbot — LangChain + Groq RAG

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY=your_real_groq_api_key
```

If you don't have Tesseract/Poppler installed, ingestion still works for PDFs
with a text layer (e.g. the Rinvoq PI) — OCR fallback just won't fire.

## Run

```bash
uvicorn app.main:app --reload
```

Then:

```bash
# ingest a PDF
curl -X POST http://localhost:8000/api/documents/upload \
  -F "drug_name=RINVOQ" \
  -F "file=@/path/to/rinvoq_pi.pdf"

# ask a question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "What is the dosage for a 25kg child with psoriatic arthritis?"}'
```

## Architecture

See `app/rag/chain.py` for the full request flow:
memory -> query rewrite -> hybrid retrieval -> rerank -> generate ->
groundedness gate (confidence-based refusal) -> safety validator gate
(hard refusal, not rewrite) -> citations.

## Known limitations / things to verify before the demo

- **BM25 index is in-memory and rebuilt per-ingestion call.** Fine for a
  single-document hackathon demo; for multiple documents, ingest all PDFs
  in one run or persist/reload the keyword corpus — otherwise ingesting a
  second drug will wipe the first drug's keyword index (vector search is
  unaffected since Chroma persists to disk).
- **Table-to-section attribution is positional** (nearest heading above the
  table by y-coordinate) — verified correct against the Rinvoq PDF's
  pediatric PsA dosing table (page 8, section 2.4), but worth spot-checking
  against any new PDF you ingest, since layout varies across brands.
- **Groundedness and safety checks each cost one extra Groq model call per
  message** (3 LLM calls total per user turn: rewrite, generate, groundedness,
  safety — actually 4). Watch Groq rate limits during a live demo.
- **Session memory is in-process and non-persistent** — restarting the
  server clears all conversations. This matches the locked-in "per-session"
  decision but is worth stating explicitly to judges.
- Multimodal image captioning only fires on images >150x150px to skip
  logos/icons — adjust the threshold in `pdf_extractor.py` if a target PDF's
  real diagrams are smaller than that.
