# mini_project.py
"""
Mini-Project: LLaMA 3.2 API with FastAPI
REST API for structured JSON generation using LLaMA 3.2-1B
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os
from dotenv import load_dotenv
import json
import uvicorn

# ----------------------------
# INITIALISATION
# ----------------------------
load_dotenv()

app = FastAPI(
    title="LLaMA 3.2 API",
    description="REST API for LLaMA 3.2-1B model with structured JSON generation",
    version="1.0.0"
)

# Authentification
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# ----------------------------
# MODÈLE & TOKENIZER (chargement lazy)
# ----------------------------
MODEL_NAME = "meta-llama/Llama-3.2-1B"
model = None
tokenizer = None

def load_model():
    """Charge le modèle LLaMA 3.2"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        print("🔄 Chargement du modèle LLaMA 3.2...")
        
        try:
            # Configuration de quantisation
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                token=HF_TOKEN if HF_TOKEN else None
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            # Modèle
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                token=HF_TOKEN if HF_TOKEN else None
            )
            
            print("✅ Modèle chargé avec succès!")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            print("Utilisation d'un modèle alternatif...")
            # Fallback
            from transformers import pipeline
            model = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            tokenizer = model.tokenizer
    
    return model, tokenizer

# ----------------------------
# SCHÉMAS PYDANTIC
# ----------------------------
class GenerationRequest(BaseModel):
    """Schéma pour les requêtes de génération"""
    prompt: str = Field(..., description="Prompt text for generation")
    max_tokens: int = Field(150, ge=1, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    json_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional JSON schema for structured output"
    )

class FactCheckRequest(BaseModel):
    """Schéma pour la vérification de faits"""
    statement: str = Field(..., description="Statement to fact-check")
    context: Optional[str] = Field(None, description="Optional context for fact-checking")

class SQLGenerationRequest(BaseModel):
    """Schéma pour la génération de SQL"""
    question: str = Field(..., description="Natural language question")
    schema_info: str = Field(..., description="Database schema information")

# ----------------------------
# FONCTIONS UTILITAIRES
# ----------------------------
def format_llama_prompt(user_message: str, system_message: str = None) -> str:
    """Formate un prompt pour LLaMA 3.2"""
    if system_message:
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}

<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

def extract_json_from_text(text: str) -> Dict:
    """Extrait et valide JSON depuis le texte généré"""
    import re
    
    # Chercher du JSON dans le texte
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Essayer de corriger les problèmes courants
            json_str = json_str.replace("'", '"').replace("True", "true").replace("False", "false")
            try:
                return json.loads(json_str)
            except:
                return {"raw_text": text, "error": "Invalid JSON generated"}
    else:
        return {"raw_text": text, "error": "No JSON found in response"}

# ----------------------------
# ENDPOINTS API
# ----------------------------
@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "LLaMA 3.2 API is running",
        "model": MODEL_NAME,
        "endpoints": [
            "/generate",
            "/generate_json",
            "/fact_check",
            "/generate_sql",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    """Vérifie l'état de santé de l'API"""
    try:
        load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_name": MODEL_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """Génération de texte standard"""
    try:
        model_obj, tokenizer_obj = load_model()
        
        # Formater le prompt
        prompt = format_llama_prompt(request.prompt)
        
        # Tokenisation
        inputs = tokenizer_obj(prompt, return_tensors="pt").to(model_obj.device)
        
        # Génération
        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=True,
                pad_token_id=tokenizer_obj.eos_token_id
            )
        
        # Décodage
        generated_text = tokenizer_obj.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire la réponse de l'assistant
        if "assistant" in generated_text:
            response = generated_text.split("assistant")[-1].strip()
        else:
            response = generated_text
        
        return {
            "prompt": request.prompt,
            "generated_text": response,
            "full_response": generated_text,
            "parameters": {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_json")
async def generate_json(request: GenerationRequest):
    """Génération de sortie JSON structurée"""
    try:
        model_obj, tokenizer_obj = load_model()
        
        # Système message pour JSON
        system_msg = """You are a precise JSON generator. 
        Respond ONLY with valid JSON matching the requested schema.
        Do not include any explanation or additional text."""
        
        # Ajouter le schéma JSON si fourni
        if request.json_schema:
            schema_str = json.dumps(request.json_schema, indent=2)
            user_msg = f"{request.prompt}\n\nJSON Schema:\n{schema_str}\n\nJSON Output:"
        else:
            user_msg = f"{request.prompt}\n\nProvide the answer as valid JSON."
        
        prompt = format_llama_prompt(user_msg, system_msg)
        
        # Tokenisation avec température plus basse pour du JSON stable
        inputs = tokenizer_obj(prompt, return_tensors="pt").to(model_obj.device)
        
        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=0.3,  # Température basse pour du JSON cohérent
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer_obj.eos_token_id
            )
        
        generated_text = tokenizer_obj.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire et valider JSON
        json_output = extract_json_from_text(generated_text)
        
        return {
            "prompt": request.prompt,
            "json_output": json_output,
            "raw_generation": generated_text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fact_check")
async def fact_check(request: FactCheckRequest):
    """Vérification de faits avec sortie JSON"""
    try:
        system_msg = """You are a fact-checking assistant. 
        Analyze statements and provide factual accuracy assessment in JSON format."""
        
        user_msg = f"""Statement to fact-check: "{request.statement}"
        
        {f"Context: {request.context}" if request.context else ""}
        
        Provide assessment in this JSON format:
        {{
            "statement": "original statement",
            "truthfulness": "true/false/partially_true",
            "confidence": "high/medium/low",
            "explanation": "brief explanation",
            "sources": ["optional sources"]
        }}"""
        
        gen_request = GenerationRequest(
            prompt=user_msg,
            max_tokens=250,
            temperature=0.3,
            json_schema={
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "truthfulness": {"type": "string"},
                    "confidence": {"type": "string"},
                    "explanation": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}}
                }
            }
        )
        
        return await generate_json(gen_request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_sql")
async def generate_sql(request: SQLGenerationRequest):
    """Génération de requêtes SQL depuis du langage naturel"""
    try:
        system_msg = """You are a SQL expert. Generate valid SQL queries from natural language questions."""
        
        user_msg = f"""Database Schema:
{request.schema_info}

Question: {request.question}

Generate a SQL query to answer this question.
Return ONLY the SQL query, no explanations."""

        gen_request = GenerationRequest(
            prompt=user_msg,
            max_tokens=200,
            temperature=0.1  # Très bas pour du SQL précis
        )
        
        result = await generate_text(gen_request)
        
        # Nettoyer la sortie SQL
        sql_query = result["generated_text"].strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return {
            "question": request.question,
            "sql_query": sql_query.strip(),
            "schema_info": request.schema_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# EXÉCUTION
# ----------------------------
if __name__ == "__main__":
    print("🚀 Démarrage de l'API LLaMA 3.2...")
    print(f"   Modèle: {MODEL_NAME}")
    print(f"   URL: http://localhost:8000")
    print(f"   Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )