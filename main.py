import os
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY manquant (Render > Environment)")

SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="DeepResearch API", version="1.0.0")

def verify_api_key(authorization: str = Header(None)):
    if not SERVICE_API_KEY:
        return True
    if not authorization or authorization != f"Bearer {SERVICE_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

class PromptRequest(BaseModel):
    prompt: str

@app.get("/healthz")
def healthz():
    return {"status": "ok", "llm_model": LLM_MODEL}

@app.post("/deep", dependencies=[Depends(verify_api_key)])
def deep(req: PromptRequest):
    messages = [
        {"role": "system", "content": "Tu es un expert RH, tu synthétises et expliques les informations pour l’utilisateur sans jamais mentionner l’IA."},
        {"role": "user", "content": req.prompt}
    ]
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1100,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur OpenAI: {str(e)}")

    return {"answer": answer}
