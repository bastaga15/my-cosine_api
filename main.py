# main.py
import os
import json
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from openai import OpenAI
import unicodedata
import requests  # pour option téléchargement depuis URL si besoin

app = FastAPI(title="Cosine Search API")

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # On laisse le déploiement échouer plus tard si clé manquante, mais on log ici
    print("⚠️ OPENAI_API_KEY non défini. Assure-toi de le configurer dans Render.")

SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")  # clé privée pour protéger l'API (ex: "mon_secret_api_key")
EMBED_JSON_PATH = os.getenv("EMBED_JSON_PATH", "etude_remunerations_2025_with_embeddings.json")
EMBED_JSON_S3_URL = os.getenv("EMBED_JSON_S3_URL")  # si tu stockes le JSON ailleurs (S3 / presigned URL)

# client OpenAI (utilise OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

# --- Helpers ---
def normalize(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()

def download_if_needed():
    """
    Si EMBED_JSON_S3_URL est défini et que le fichier local n'existe pas,
    on le télécharge (utile pour gros fichiers stockés hors repo).
    """
    if EMBED_JSON_S3_URL and not os.path.exists(EMBED_JSON_PATH):
        print(f"Téléchargement du fichier d'embeddings depuis {EMBED_JSON_S3_URL} …")
        r = requests.get(EMBED_JSON_S3_URL)
        r.raise_for_status()
        with open(EMBED_JSON_PATH, "wb") as f:
            f.write(r.content)
        print("Téléchargement terminé.")

# Charger et prétraiter embeddings au démarrage
print("Chargement des embeddings …")
download_if_needed()
if not os.path.exists(EMBED_JSON_PATH):
    raise RuntimeError(f"Fichier d'embeddings introuvable: {EMBED_JSON_PATH}")

with open(EMBED_JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Préparer matrice numpy normalisée pour accélérer les dot-products
emb_list = []
for item in data:
    emb = np.array(item.get("embedding", []), dtype=np.float32)
    if emb.size == 0:
        # si un item n'a pas d'embedding, on met un vecteur nul
        emb = np.zeros((1536,), dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        emb_norm = emb
    else:
        emb_norm = emb / norm
    emb_list.append(emb_norm)

embedding_matrix = np.stack(emb_list)  # shape (N, dim)
print(f"Embeddings loaded: {len(emb_list)} items, dim={embedding_matrix.shape[1]}")

# Liste des régions possibles (si présentes dans les items)
regions_possibles = []
if len(data) > 0 and isinstance(data[0].get("regions", {}), dict):
    # récupère toutes les clés de regions du premier item (supposées cohérentes)
    regions_possibles = list(data[0]["regions"].keys())

# API key verify (Authorization: Bearer <SERVICE_API_KEY>) si SERVICE_API_KEY est défini
def verify_api_key(authorization: Optional[str] = Header(None)):
    if SERVICE_API_KEY:
        if not authorization or authorization != f"Bearer {SERVICE_API_KEY}":
            raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# Schemas
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    detect_region: Optional[bool] = True

class ResultItem(BaseModel):
    score: float
    text: str
    salaire_base: str
    salaire_corrige: List[float]
    pond: str
    region: Optional[str]

# Endpoint health
@app.get("/health")
def health():
    return {"status": "ok", "items": len(data)}

# Endpoint de recherche
@app.post("/search", dependencies=[Depends(verify_api_key)])
def search(req: SearchRequest):
    query = req.query
    top_k = req.top_k or 3

    # 1) embedding de la requête
    try:
        emb_resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        query_emb = np.array(emb_resp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur embeddings: {str(e)}")

    qnorm = query_emb / (np.linalg.norm(query_emb) + 1e-12)

    # 2) detection de la région si demandé
    region_recherche = None
    if req.detect_region and regions_possibles:
        qnorm_small = normalize(query)
        for r in regions_possibles:
            if normalize(r) in qnorm_small:
                region_recherche = r
                break
        # (optionnel) on ne met rien si aucune région détectée

    # 3) calcul scores (dot product = cosine between normalized vectors)
    scores = embedding_matrix.dot(qnorm)  # shape (N,)
    top_idx = np.argsort(scores)[-top_k:][::-1]  # indices des top_k

    results = []
    for idx in top_idx:
        item = data[int(idx)]
        score = float(scores[int(idx)])
        # pondération régionale
        regions = item.get("regions", {})
        pond = regions.get(region_recherche, "0%") if region_recherche else "0%"

        # calcul salaire corrigé (comme ton code)
        salaire_base = item.get("salaire_k€", "0-0")
        try:
            bas, haut = [float(x.strip()) for x in salaire_base.split("-")]
        except:
            bas, haut = 0.0, 0.0

        if isinstance(pond, str) and "%" in pond:
            if "à" in pond:
                pond_val = -0.05
            else:
                try:
                    pond_val = float(pond.replace("%", "").replace(",", ".").strip()) / 100
                except:
                    pond_val = 0.0
        else:
            pond_val = 0.0

        salaire_corrige = [bas * (1 + pond_val), haut * (1 + pond_val)]

        results.append({
            "score": score,
            "text": item.get("text", "")[:1000],  # tronque éventuellement
            "salaire_base": salaire_base,
            "salaire_corrige": [round(s, 2) for s in salaire_corrige],
            "pond": pond,
            "region": region_recherche
        })

    return {"status": "ok", "query": query, "region": region_recherche, "results": results}
