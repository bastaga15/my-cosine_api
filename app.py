# app.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from openai import OpenAI

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent
FAMILIES_DIR = BASE_DIR / "familles"
LABEL_META_FILE = FAMILIES_DIR / "file_labels_meta.json"

# IMPORTANT: tu as indexé avec "text-embedding-3-large" (3072 dims)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY manquant (Render > Environment)")

SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")  # pour sécuriser /search

client = OpenAI(api_key=OPENAI_API_KEY)

TEXT_KEYS = ("texte", "text", "content", "body")
# Clés possibles pour les embeddings item-level (pré-calculés)
EMB_KEYS = (
    "embedding", "embeddings", "vector", "embedding_vector",
    "openai_embedding",
)

app = FastAPI(title="Familles Search API (OpenAI)", version="1.0.0")


# ---------------- Helpers ----------------
def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, eps)

def embed_openai(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    X = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (X / norms).astype(np.float32)

def encode_query(q: str) -> np.ndarray:
    return embed_openai([q])[0]  # (d,)

def cosine_scores(q_vec: np.ndarray, M: np.ndarray) -> np.ndarray:
    return M @ q_vec  # cosines (si M et q_vec sont normalisés)


# ---------------- Auth (API key) ----------------
def verify_api_key(authorization: Optional[str] = Header(None)):
    if not SERVICE_API_KEY:
        # Pas d'API key configurée -> route publique
        return True
    if not authorization or authorization != f"Bearer {SERVICE_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# --------- Chargement des labels pour le routing ----------
if not LABEL_META_FILE.exists():
    raise FileNotFoundError(f"Fichier introuvable: {LABEL_META_FILE}")

with LABEL_META_FILE.open("r", encoding="utf-8") as f:
    FILE_LABELS_META: List[Dict[str, Any]] = json.load(f)

FILENAMES = [e["filename"] for e in FILE_LABELS_META]
FILEPATHS = [e["file"] for e in FILE_LABELS_META]
LABEL_TEXTS = [
    e.get("text_for_embedding")
    or e.get("label")
    or Path(e["file"]).stem.replace("_", " ")
    for e in FILE_LABELS_META
]

# Embeddings des labels (routing) au démarrage
LABELS_EMB = embed_openai(LABEL_TEXTS)  # (N, d)
LABEL_DIM = LABELS_EMB.shape[1]


# ---------------- Extraction embeddings item-level ----------------
def pick_embedding(item: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Extrait l'embedding d'un item si présent (liste brute ou dict).
    Retourne un vecteur L2-normalisé ou None.
    """
    for k in EMB_KEYS:
        if k in item:
            v = item[k]
            # Liste brute
            if isinstance(v, list) and v and isinstance(v[0], (float, int)):
                arr = np.asarray(v, dtype=np.float32)
                return l2_normalize(arr)
            # Dict avec 'embedding'/'vector'/'values'/'data'
            if isinstance(v, dict):
                for kk in ("embedding", "vector", "values", "data"):
                    w = v.get(kk)
                    if isinstance(w, list) and w and isinstance(w[0], (float, int)):
                        arr = np.asarray(w, dtype=np.float32)
                        return l2_normalize(arr)
    return None


# ---------------- Routing fichiers + recherche fine ----------------
def route_files(query: str, top_k: int) -> List[Dict[str, Any]]:
    q = encode_query(query)
    scores = cosine_scores(q, LABELS_EMB)
    order = np.argsort(-scores)[:top_k]
    routed = []
    for i in order:
        routed.append({
            "filename": FILENAMES[i],
            "file": FILEPATHS[i],
            "label": LABEL_TEXTS[i],
            "score": float(scores[i]),
        })
    return routed

def search_in_file(query_vec: np.ndarray, file_path: str, top_n: int) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Recherche dans un fichier en utilisant UNIQUEMENT les embeddings stockés dans les items.
    Les items sans embedding sont ignorés. Aucun recalcul à la volée.
    """
    fp = Path(file_path)
    if not fp.exists():
        return []

    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Support d'un wrapper {"items": [...]}
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        data = data["items"]
    if not isinstance(data, list):
        return []

    embs = []
    items = []
    skipped_dim = 0

    for it in data:
        e = pick_embedding(it)
        if e is None:
            continue
        # Vérifie la compat dimension (doit matcher text-embedding-3-large si c'est ton modèle)
        if e.shape[0] != query_vec.shape[0]:
            skipped_dim += 1
            continue
        embs.append(e)
        items.append(it)

    if not embs:
        return []

    M = np.vstack(embs)  # (N, d) normalisés
    scores = M @ query_vec
    order = np.argsort(-scores)[:top_n]
    return [(float(scores[i]), items[i]) for i in order]


# ---------------- Schémas API ----------------
class SearchRequest(BaseModel):
    query: str
    top_files: int = 3
    top_results: int = 10


# ---------------- Endpoints ----------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "files": len(FILE_LABELS_META),
        "embedding_model": EMBEDDING_MODEL,
        "label_dim": LABEL_DIM,
    }

@app.post("/search", dependencies=[Depends(verify_api_key)])
def search(req: SearchRequest):
    # 1) Routing des fichiers par labels
    routed = route_files(req.query, req.top_files)

    # 2) Embedding de la requête
    try:
        q = encode_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur embeddings OpenAI: {str(e)}")

    # 3) Recherche fine (items avec embeddings)
    all_hits: List[Tuple[float, Dict[str, Any], str]] = []
    per_file_top = max(10, req.top_results)
    for r in routed:
        hits = search_in_file(q, r["file"], per_file_top)
        for s, it in hits:
            all_hits.append((s, it, r["filename"]))

    all_hits.sort(key=lambda x: -x[0])
    top = all_hits[:req.top_results]

    results = []
    for score, item, fname in top:
        snippet = (item.get("texte") or item.get("text") or "")[:200].replace("\n", " ")
        meta = {k: item.get(k) for k in ("famille", "poste", "region", "anciennete") if k in item}
        results.append({
            "score": score,
            "filename": fname,
            "meta": meta,
            "snippet": snippet,
            "raw": item,  # enlève si tu ne veux pas renvoyer la donnée brute
        })

    return {
        "files_routed": routed,
        "results": results,
    }
