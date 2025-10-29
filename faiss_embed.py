import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_or_update_faiss(json_file, faiss_dir="faiss_index", model_name="all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = list(records.values())

    texts = [r["document"] for r in records]
    metadatas = [{**r["metadata"], "id": r["id"]} for r in records]

    try:
        db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)

        # Collect existing IDs
        existing_ids = {doc.metadata.get("id") for doc in db.similarity_search("", k=len(db.docstore._dict))}

        # Filter out duplicates
        new_data = [(t, m) for t, m in zip(texts, metadatas) if m["id"] not in existing_ids]

        if new_data:
            new_texts, new_metas = zip(*new_data)
            new_db = FAISS.from_texts(list(new_texts), embeddings, metadatas=list(new_metas))
            db.merge_from(new_db)

    except Exception:
        db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    db.save_local(faiss_dir)
    return {"status": "Success", "added": len(texts), "saved_to": faiss_dir}

if __name__ == "__main__":
    build_or_update_faiss("fpend_data.json")
    build_or_update_faiss("fst_data.json")
