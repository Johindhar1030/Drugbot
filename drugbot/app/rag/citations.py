"""Citation extraction and formatting.

Every citation object returned to the frontend has:
  document      – drug name / document title
  section       – section number + title (never "UNSPECIFIED")
  page          – page number or "Not available"
  chunk_ref     – internal reference
"""
import re


def clean_document_name(raw_name: str | None) -> str:
    """Format raw drug name or PDF filename into a clean document title.
    e.g. 'skyrizi_pi.pdf' -> 'SKYRIZI Prescribing Information'
         'rinvoq' -> 'RINVOQ Prescribing Information'
         'SKYRIZI' -> 'SKYRIZI Prescribing Information'
    """
    if not raw_name:
        return "Prescribing Information"
    s = raw_name.strip()
    if s.lower().endswith(".pdf"):
        s = s[:-4].strip()
    if s.lower().endswith("_pi") or s.lower().endswith("-pi"):
        s = s[:-3].strip()
    if s.lower().endswith("_label") or s.lower().endswith("-label"):
        s = s[:-6].strip()

    words = s.replace("_", " ").replace("-", " ").split()
    if not words:
        return "Prescribing Information"

    cleaned_words = []
    for w in words:
        if len(w) <= 6 or w.upper() in ("SKYRIZI", "RINVOQ", "HUMIRA", "STELARA", "DUPIXENT", "BRENZYS"):
            cleaned_words.append(w.upper())
        else:
            cleaned_words.append(w.capitalize())

    title = " ".join(cleaned_words)
    if "Prescribing Information" not in title and "PI" not in title:
        title += " Prescribing Information"
    return title


def _clean_section(raw: str | None) -> str:
    """Return a display-safe section string; never return UNSPECIFIED."""
    if not raw or raw.strip().upper() in ("UNSPECIFIED", "", "NONE"):
        return "Not available"
    return raw.strip()


def _abbreviate_section_name(section: str) -> str:
    """Abbreviates lengthy FDA section headings for compact inline citation display.
    e.g. '17 PATIENT COUNSELING INFORMATION' -> '§17'
         '2.1 Recommended Dosage and Administration' -> '§2.1'
         'HIGHLIGHTS OF PRESCRIBING INFORMATION' -> 'Highlights'
    """
    if not section or section == "Not available":
        return section

    s = section.strip()
    # Match numbered section e.g. "17 PATIENT COUNSELING INFORMATION" -> "§17"
    m_num = re.match(r"^§?\s*(\d{1,2}(?:\.\d{1,2})?)(?:\s+.*)?$", s)
    if m_num:
        return f"§{m_num.group(1)}"

    upper_s = s.upper()
    if "HIGHLIGHTS" in upper_s:
        return "Highlights"
    if "BOXED WARNING" in upper_s:
        return "Boxed Warning"
    if "MEDICATION GUIDE" in upper_s:
        return "Med Guide"
    if "INSTRUCTIONS FOR USE" in upper_s or "IFU" in upper_s:
        return "IFU"
    if "PATIENT INFORMATION" in upper_s or "PATIENT PACKAGE INSERT" in upper_s:
        return "Patient Info"
    if "PATIENT COUNSELING" in upper_s:
        return "§17"

    return f"§{s}" if not s.startswith("§") else s


def _format_single_source(meta: dict, compact: bool = True) -> str:
    section_raw = meta.get("section") or ""
    section = _clean_section(section_raw)
    if compact and section != "Not available":
        section = _abbreviate_section_name(section)
    page = meta.get("page_number")

    parts = []
    if section and section != "Not available":
        if not section.startswith("§") and not section.lower().startswith("section") and not section in ("Highlights", "Boxed Warning", "Med Guide", "IFU", "Patient Info"):
            parts.append(f"§{section}")
        else:
            parts.append(section)
    if page and page not in (None, "Not available", "N/A"):
        parts.append(f"p.{page}")

    if parts:
        return ", ".join(parts)
    return clean_document_name(meta.get("drug_name"))


def extract_citations(answer_text: str, chunks: list[dict]) -> list[dict]:
    """Resolve every chunk reference in the answer to deduplicated citation objects.

    Deduplicates by (document, section, page) so that multiple chunks pointing to the
    same source location produce only one user-facing citation entry.
    Internal chunk_ref is preserved in each citation for debugging/tracing.
    """
    used_indices = sorted(
        set(int(m) for m in re.findall(r"chunk_(\d+)", answer_text, re.IGNORECASE))
    )
    if not used_indices and chunks:
        # Fallback to top retrieved context chunks if no explicit inline markers were placed
        used_indices = list(range(min(3, len(chunks))))

    citations = []
    seen_keys = set()

    for idx in used_indices:
        if idx < 0 or idx >= len(chunks):
            continue
        meta = chunks[idx].get("metadata", {})
        doc = clean_document_name(meta.get("drug_name"))
        section = _clean_section(meta.get("section") or "")
        page = meta.get("page_number") or "Not available"

        # Deduplicate by (document, section, page)
        dedup_key = (doc, section, str(page))
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        citations.append({
            "chunk_ref": f"chunk_{idx}",
            "document": doc,
            "section": section,
            "page": page,
            "source_label": _format_single_source(meta),
        })
    return citations


def replace_chunk_markers_with_sources(answer_text: str, chunks: list[dict]) -> str:
    """Replace all [chunk_N], 【chunk_N】, (chunk_N), etc. with human-readable PDF source citations:
    e.g. '[§BOXED WARNING, p.1]' or '[§4 CONTRAINDICATIONS, p.2]'."""
    def _replacer(match: re.Match) -> str:
        content = match.group(0)
        indices = [int(x) for x in re.findall(r"chunk_(\d+)", content, re.IGNORECASE)]
        if not indices:
            return ""

        sources = []
        seen = set()
        for idx in indices:
            if 0 <= idx < len(chunks):
                meta = chunks[idx].get("metadata", {})
                src = _format_single_source(meta)
                if src and src not in seen:
                    seen.add(src)
                    sources.append(src)

        if sources:
            return f" [{'; '.join(sources)}]"
        return ""

    # Match bracketed chunk references (ASCII [], fullwidth 【】, parens ()) or bare chunk_N
    pattern = re.compile(
        r"(?:\[|\【|\(\[?)(?:[^\}\]\>\)\】]*\bchunk_\d+\b[^\}\]\>\)\】]*)(?:\]|\】|\)\]?)"
        r"|\bchunk_\d+\b",
        re.IGNORECASE,
    )
    res = pattern.sub(_replacer, answer_text)
    # Clean up double spaces or space before punctuation
    res = re.sub(r"\s+([.,;:!?])", r"\1", res)
    res = re.sub(r" +", " ", res)
    return res.strip()


def sanitize_response_text(text: str) -> str:
    """Safety filter ensuring NO internal chunk identifiers (chunk_0, chunk_1, 【chunk_0】, etc.) leak to the user.

    Strips raw chunk identifiers while strictly preserving medical terminology, drug names, section names,
    page numbers, citations, and evidence labels.
    """
    if not text:
        return ""

    res = text
    # 1. Remove bracketed chunk markers e.g. [chunk_0], 【chunk_0】, (chunk_0), [[chunk_0]]
    res = re.sub(
        r"(?:\[|\【|\(\[?)\s*chunk_\d+(?:\s*,\s*chunk_\d+)*\s*(?:\]|\】|\)\]?)",
        "",
        res,
        flags=re.IGNORECASE,
    )

    # 2. Remove bare or leftover chunk_\d+ occurrences
    res = re.sub(r"\bchunk_\d+\b", "", res, flags=re.IGNORECASE)

    # 3. Clean up empty or broken brackets left behind: [], 【】, (), [  ]
    res = re.sub(r"\[\s*\]", "", res)
    res = re.sub(r"【\s*】", "", res)
    res = re.sub(r"\(\s*\)", "", res)

    # 4. Clean up spacing and punctuation glitches caused by removal
    res = re.sub(r"[ \t]+([.,;:!?])", r"\1", res)
    res = re.sub(r"\n{3,}", "\n\n", res)
    res = re.sub(r"[ \t]+", " ", res)

    return res.strip()


def strip_citation_markers(answer_text: str) -> str:
    """Strip all raw chunk markers and bracketed chunk citations from the text."""
    pattern = re.compile(
        r"\s*(?:\[|\【|\(\[?)(?:[^\}\]\>\)\】]*\bchunk_\d+\b[^\}\]\>\)\】]*)(?:\]|\】|\)\]?)"
        r"|\s*\bchunk_\d+\b",
        re.IGNORECASE,
    )
    res = pattern.sub("", answer_text)
    res = re.sub(r"\s+([.,;:!?])", r"\1", res)
    res = re.sub(r" +", " ", res)
    return res.strip()


def format_citations_text(citations: list[dict]) -> str:
    """Produce a plain-text source block for embedding in the answer when needed."""
    if not citations:
        return ""
    lines = []
    seen = set()
    for c in citations:
        doc = clean_document_name(c.get("document"))
        sec = c.get("section")
        page = c.get("page")
        key = (doc, sec, page)
        if key in seen:
            continue
        seen.add(key)
        
        c_lines = [f"**Source:** {doc}"]
        if sec and sec not in ("Not available", "N/A", "UNSPECIFIED"):
            c_lines.append(f"**Section:** {sec}")
        if page and page not in ("Not available", "N/A"):
            c_lines.append(f"**Page:** {page}")
        lines.append("\n".join(c_lines))

    return "\n\n".join(lines)


def format_structured_citation_block(citations: list[dict]) -> str:
    """Produce clean Markdown citation block:
    
    **Source:** SKYRIZI Prescribing Information
    **Section:** Medication Guide
    **Page:** 43
    """
    return format_citations_text(citations)

