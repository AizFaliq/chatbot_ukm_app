import re
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import lmstudio as lms  # LM Studio client

# --- Prompt ---
PROMPT_TEMPLATE = """
Anda ialah pembantu yang hanya menjawab dalam Bahasa Melayu.
Gunakan maklumat dalam konteks di bawah untuk membina jawapan.
Jangan ulang soalan. Jangan tambah atau memaparkan "Q:" atau "A:" dalam apa jua keadaan.

Peraturan tambahan untuk jawapan:
1. Jika jawapan mengandungi **lebih daripada satu perkara, langkah, contoh atau ciri**, susun **dalam bentuk senarai bernombor** supaya lebih mudah dibaca dan difahami.
2. Jika soalan bersifat **"ya/tidak"** (contohnya bermula dengan "Adakah..."), sertakan jawapan **"ya" atau "tidak"** berserta **penjelasan ringkas** untuk menyokongnya.
3. Jika soalan berbentuk **terbuka** (contohnya bermula dengan "Apakah", "Bagaimanakah", atau "Mengapakah"), **jangan mulakan jawapan dengan "ya" atau "tidak"** — berikan terus penjelasan yang relevan.
4. Jika jawapan terlalu ringkas atau terlalu umum, berikan **penjelasan tambahan** secara ringkas untuk memperkukuh pemahaman.
5. Jawapan mesti dalam **ayat lengkap**, disampaikan secara **jelas, teratur**, dan **sesuai dengan bentuk soalan**.

Ikuti arahan pemformatan ini secara ketat:
{rendering_rules}

Konteks:
{context}

Soalan: {question}
Jawapan:
"""


# --- Heuristics (controller outside the prompt) ---
EXPLAIN_TRIGGERS = re.compile(r'^(mengapa|kenapa|jelaskan|huraikan|terangkan)\b', re.I)

def detect_q_type(q: str) -> str:
    ql = q.strip().lower()
    if re.match(r'^(adakah|betulkah|apakah benar)\b', ql): return "YN"
    if EXPLAIN_TRIGGERS.match(ql): return "EXPLAIN"
    if re.search(r'\b(nyatakan|senaraikan|apakah (jenis|bentuk|kategori))\b', ql): return "LIST"
    if re.match(r'^(apakah|bagaimanakah|apakah maksud)\b', ql): return "OPEN"
    return "OPEN"

def split_candidates(text: str):
    parts = re.split(r'[.;,\n]+', text)
    parts = [p.strip() for p in parts if p.strip()]
    parts = clean_tokens(parts)
    return parts

def is_listy_text(text: str) -> bool:
    items = split_candidates(text)
    if len(items) < 3: return False
    short = sum(1 for t in items if len(t.split()) <= 3)
    return short / len(items) >= 0.7

def decide_output_mode(q_type: str, ctx_text: str) -> str:
    if q_type == "YN": return "yesno"
    if q_type == "LIST" or is_listy_text(ctx_text): return "list"
    return "paragraph"

def rendering_directive(output_mode: str) -> str:
    if output_mode == "yesno":
        return "Mulakan dengan 'Ya' atau 'Tidak' dan sertakan sebab ringkas. Jangan gunakan senarai."
    if output_mode == "list":
        # Make it STRICT
        return "HASILKAN HANYA SENARAI BERNOMBOR. SATU ITEM SATU BARIS. TIADA PERENGGAN PANJANG."
    return "Jawab dalam satu atau dua perenggan yang jelas dan teratur. Elakkan senarai kecuali benar-benar perlu."

def format_numbered(items):
    # de-dup while preserving order
    seen, out = set(), []
    for it in items:
        key = it.lower()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return "\n".join(f"{i+1}. {it}" for i, it in enumerate(out))

def hard_list_from_context(question: str, context_text: str):
    """Deterministic path: if context is listy and question not asking to explain, render a clean list."""
    if EXPLAIN_TRIGGERS.match(question.strip().lower()):
        return None  # explanation requested -> let LLM handle
    items = split_candidates(context_text)
    # Keep only short-ish tokens (avoid full sentences)
    items = [t for t in items if len(t.split()) <= 5]
    if len(items) >= 3:
        return format_numbered(items)
    return None

# --- FAISS ---
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# --- Post-processing enforcement ---
def enforce_output_mode(answer: str, output_mode: str, fallback_items=None) -> str:
    if output_mode != "list":
        return answer.strip()

    # If list requested but got paragraph, try to extract items
    lines = [ln.strip("-• \t") for ln in answer.strip().splitlines() if ln.strip()]
    looks_like_list = all(re.match(r'^(\d+\.|[-•])\s', ln) or (len(ln.split()) <= 5) for ln in lines) and len(lines) >= 3
    if looks_like_list:
        # Normalize to numbered
        items = []
        for ln in lines:
            ln = re.sub(r'^(\d+\.|[-•])\s*', '', ln).strip()
            if ln: items.append(ln)
        return format_numbered(items)

    # Try splitting the paragraph into short terms
    parts = split_candidates(answer)
    parts = [p for p in parts if len(p.split()) <= 5]
    if len(parts) >= 3:
        return format_numbered(parts)

    # Last resort: use items extracted from context (if provided)
    if fallback_items:
        return format_numbered(fallback_items)

    return answer.strip()

QA_LINE = re.compile(r'^\s*(soalan|jawapan)\s*:\s*', re.I)

def clean_tokens(items):
    cleaned = []
    for t in items:
        t = t.strip()
        if not t or re.fullmatch(r'[-–—]+', t):  # drop '---' etc.
            continue
        t = QA_LINE.sub('', t)  # drop "Soalan:" / "Jawapan:"
        if t.lower() in {"soalan", "jawapan", "konteks", "rujukan"}:
            continue
        cleaned.append(t)
    return cleaned

def extract_answer_terms_from_doc(page_text):
    """
    Prefer terms after 'Jawapan:' in the PRIMARY doc.
    Split by commas/semicolons/newlines, keep short noun-ish chunks.
    """
    m = re.search(r'jawapan\s*:\s*(.*)', page_text, flags=re.I|re.S)
    if not m:
        return []
    ans_block = m.group(1)
    # stop at next 'Soalan:' if present
    ans_block = re.split(r'\bSoalan\s*:', ans_block, flags=re.I)[0]
    parts = re.split(r'[,\n;]+', ans_block)
    parts = [p.strip() for p in parts if p.strip()]
    parts = clean_tokens(parts)
    # keep compact items (<= 4 words)
    parts = [p for p in parts if 1 <= len(p.split()) <= 4]
    # title-case single nouns like "kubus", keep acronyms as-is
    norm = []
    for p in parts:
        if p.isupper():
            norm.append(p)
        else:
            norm.append(p.capitalize())
    # de-dup preserve order
    seen, out = set(), []
    for it in norm:
        key = it.lower()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

def query_rag(query_text: str, db, model, k=5, score_threshold=0.6, max_docs=2):
    results = db.similarity_search_with_score(query_text, k=k)
    if not results:
        return ("Maaf, tiada rujukan ditemui untuk soalan ini dalam pangkalan ilmu.", [])

    # lower score = better (cosine distance)
    results = sorted(results, key=lambda x: x[1])

    filtered = [(doc, score) for doc, score in results if score <= score_threshold]
    if not filtered:
        filtered = [results[0]]  # fallback to top-1
    filtered = filtered[:max_docs]

    # PRIMARY doc used for deterministic extraction
    primary_doc, primary_score = filtered[0]

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered])

    # Heuristics
    q_type = detect_q_type(query_text)
    output_mode = decide_output_mode(q_type, context_text)
    rules = rendering_directive(output_mode)

    # 🔒 Deterministic path for "apakah ... jenis|bentuk|kategori"
    ql = query_text.strip().lower()
    wants_terms = (
        ql.startswith("apakah")
        and ((" jenis" in ql) or (" bentuk" in ql) or (" kategori" in ql))
    )

    if output_mode == "list" and wants_terms:
        terms = extract_answer_terms_from_doc(primary_doc.page_content)
        if len(terms) >= 3:
            answer_text = format_numbered(terms)
            sources = []
            doc_id = primary_doc.metadata.get("id") or primary_doc.metadata.get("source") or "[no-id]"
            sources.append(f"{doc_id} (score={primary_score:.3f})")
            return answer_text, sources
        # else: fall through to model path

    # Deterministic fast-list if context is listy
    direct_list = None
    if output_mode == "list":
        direct_list = hard_list_from_context(query_text, context_text)

    if direct_list:
        answer_text = direct_list
    else:
        chat_prompt = PROMPT_TEMPLATE.format(
            context=context_text,
            question=query_text,
            rendering_rules=rules
        )
        response = model.respond(chat_prompt)  # your client call
        answer_text = getattr(response, "content", str(response))

        if output_mode == "list":
            fallback_items = [t for t in split_candidates(context_text) if len(t.split()) <= 5]
            answer_text = enforce_output_mode(answer_text, "list", fallback_items=fallback_items)

    # Sources
    sources = []
    for doc, score in filtered:
        doc_id = doc.metadata.get("id") or doc.metadata.get("source") or "[no-id]"
        sources.append(f"{doc_id} (score={score:.3f})")

    return answer_text.strip(), sources


# --- Main loop ---
if __name__ == "__main__":
    db = load_db()
    model = lms.llm()

    print("📚 RAG Chatbot (FAISS + LM Studio)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("❓ Question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        answer, sources = query_rag(query, db, model)
        print("\n✅ Answer:\n" + answer)
        print("📖 Sources:")
        for s in sources:
            print("   •", s)
        print()
