# route_and_search.py
import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

# --------- CONFIG ---------
DIR = "familles"
LABEL_META = os.path.join(DIR, "file_labels_meta.json")
TOP_FILES = 3       # nb de fichiers à ouvrir après routing
TOP_RESULTS = 10    # nb de résultats finaux
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_KEYS = ("texte", "text", "content", "body")
EMB_KEYS  = ("embedding", "embeddings", "vector", "embedding_vector")
# --------------------------

model = SentenceTransformer(MODEL_NAME)

def l2_norm(x: np.ndarray, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / max(n, eps)

def encode(texts):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def cosine_scores(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    # q: (d,), M: (N,d) déjà normalisés
    return M @ q

def load_label_meta() -> List[Dict[str, Any]]:
    with open(LABEL_META, "r", encoding="utf-8") as f:
        return json.load(f)

def route_files(query: str, top_k: int = TOP_FILES) -> List[Dict[str, Any]]:
    meta = load_label_meta()
    if not meta:
        print("⚠ Aucun label dans file_labels_meta.json")
        return []
    q = encode([query])[0]
    M = np.array([m["embedding"] for m in meta], dtype=np.float32)  # (N, d) normalisé
    scores = cosine_scores(q, M)
    order = np.argsort(-scores)[:top_k]
    routed = []
    for i in order:
        routed.append({**meta[i], "route_score": float(scores[i])})
    print("Fichiers retenus:")
    for r in routed:
        print(f"- {r['filename']} (score={r['route_score']:.3f})")
    return routed

def pick_embedding(item):
    # Essaie de récupérer un embedding s’il existe
    for k in EMB_KEYS:
        if k in item:
            v = item[k]
            if isinstance(v, list) and v and isinstance(v[0], (float, int)):
                return np.array(v, dtype=np.float32)
            if isinstance(v, dict):
                for kk in ("values", "data"):
                    if kk in v and isinstance(v[kk], list) and v[kk] and isinstance(v[kk][0], (float, int)):
                        return np.array(v[kk], dtype=np.float32)
    return None

def pick_text(item):
    for k in TEXT_KEYS:
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k]
    return None

def search_in_file(query_vec: np.ndarray, file_path: str, top_n: int) -> List[Tuple[float, Dict[str, Any]]]:
    if not os.path.exists(file_path):
        # Le fichier peut ne pas encore exister; on ignore
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        return []

    embs_existing = []
    items_existing = []
    texts_missing = []
    idx_missing = []

    for i, it in enumerate(data):
        e = pick_embedding(it)
        if e is not None:
            # Normalise si pas normalisé
            e = l2_norm(e.astype("float32"))
            embs_existing.append(e)
            items_existing.append(it)
        else:
            t = pick_text(it)
            if t:
                texts_missing.append(t)
                idx_missing.append(i)

    mats = []
    merged_items = []
    if embs_existing:
        mats.append(np.vstack(embs_existing))
        merged_items.extend(items_existing)
    if texts_missing:
        enc = encode(texts_missing)  # déjà normalisé
        mats.append(enc)
        merged_items.extend([data[j] for j in idx_missing])

    if not mats:
        return []

    M = np.vstack(mats)  # (N, d)
    scores = M @ query_vec  # cosines

    top_idx = np.argsort(-scores)[:top_n]
    return [(float(scores[j]), merged_items[j]) for j in top_idx]

def search(query: str, top_files: int = TOP_FILES, top_results: int = TOP_RESULTS):
    q = encode([query])[0]  # (d,)
    routed = route_files(query, top_files)

    all_hits: List[Tuple[float, Dict[str, Any], str]] = []
    per_file_top = max(10, top_results)
    for r in routed:
        hits = search_in_file(q, r["file"], per_file_top)
        for s, it in hits:
            all_hits.append((s, it, r["filename"]))

    all_hits.sort(key=lambda x: -x[0])
    return all_hits[:top_results]

if _name_ == "_main_":
    query = "Téléconseiller banque de détail en ligne en région Grand Est 3 ans d'ancienneté"
    results = search(query, top_files=3, top_results=10)
    print("\nTop résultats:")
    for score, item, fname in results:
        txt = (item.get("texte") or item.get("text") or "")[:140].replace("\n", " ")
        meta = {k: item.get(k) for k in ("famille", "poste", "region", "anciennete") if k in item}
        print(f"- {score:.3f} | {fname} | {meta} | {txt}...")
