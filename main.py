# main.py
import os
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY manquant (Render > Environment)")

SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")  # clé privée pour protéger l'API (ex: "mon_secret_api_key")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="DeepResearch API", version="1.0.0")

def verify_api_key(authorization: str = Header(None)):
    if not SERVICE_API_KEY:
        return True
    if not authorization or authorization != f"Bearer {SERVICE_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

class DeepRequest(BaseModel):
    query: str
    items: list  # liste reçue depuis familles-search en entrée
    instruction: str = "Synthétise la donnée la plus pertinente pour l’utilisateur."

@app.get("/healthz")
def healthz():
    return {"status": "ok", "llm_model": LLM_MODEL}

@app.post("/deep", dependencies=[Depends(verify_api_key)])
def deep(req: DeepRequest):
    # Construction du prompt
    prompt = (
        f"{req.instruction}\n"
        f"Requête utilisateur: {req.query}\n"
        f"Voici les données extraites :\n\n"
    )
    for i, item in enumerate(req.items):
        snippet = item.get("snippet") or item.get("text") or ""
        meta = item.get("meta") or {}
        prompt += f"- {snippet} ({meta})\n"

    messages = [
        {"role": "system", "content": "Tu es un expert RH, tu synthétises et expliques les informations pour l’utilisateur."},
        {"role": "user", "content": prompt}
    ]
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=900,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur OpenAI: {str(e)}")

    return {"answer": answer}
