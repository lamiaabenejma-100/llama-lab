# mini_project.py
"""
Mini-Project: API avec demande interactive du token
"""

# =============== AJOUTEZ CES LIGNES POUR FIX SSL ===============
import os
import ssl
import warnings

# Fix pour les problèmes de certificat SSL
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
warnings.filterwarnings("ignore")
# ================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import json
import uvicorn
import sys

# ----------------------------
# FONCTION POUR DEMANDER LE TOKEN AU DÉMARRAGE
# ----------------------------
def get_hf_token_on_startup():
    """Demande le token au démarrage de l'API"""
    
    print("\n" + "=" * 60)
    print("🔐 CONFIGURATION DE L'API")
    print("=" * 60)
    
    HF_TOKEN = None
    USE_LLAMA = False
    
    # Essayer d'abord .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if HF_TOKEN:
            print("✅ Token trouvé dans .env")
            USE_LLAMA = True
    except:
        pass
    
    # Sinon demander
    if not HF_TOKEN:
        print("\nPour utiliser LLaMA 3.2, un token Hugging Face est optionnel.")
        print("Sans token, l'API utilisera un modèle open-source.")
        
        choice = input("\nVoulez-vous entrer un token? (o/N): ").strip().lower()
        
        if choice == 'o' or choice == 'oui':
            print("\n" + "-" * 40)
            print("INSTRUCTIONS:")
            print("1. Allez sur: https://huggingface.co/settings/tokens")
            print("2. Créez un token 'read'")
            print("3. Copiez le token (commence par 'hf_')")
            print("-" * 40)
            
            token = input("\nEntrez votre token (laisser vide pour modèle open-source): ").strip()
            
            if token and token.startswith('hf_'):
                HF_TOKEN = token
                USE_LLAMA = True
                
                # Sauvegarder optionnel
                save = input("\nSauvegarder dans .env? (o/N): ").strip().lower()
                if save == 'o' or save == 'oui':
                    with open(".env", "w") as f:
                        f.write(f"HUGGING_FACE_HUB_TOKEN={token}")
                    print("✅ Token sauvegardé")
    
    # Authentification si token disponible
    if USE_LLAMA and HF_TOKEN:
        try:
            login(token=HF_TOKEN)
            print("✅ Authentification Hugging Face réussie")
        except Exception as e:
            print(f"⚠️  Erreur d'authentification: {e}")
            print("Utilisation d'un modèle open-source...")
            USE_LLAMA = False
    else:
        print("✅ Utilisation d'un modèle open-source")
    
    return HF_TOKEN, USE_LLAMA

# ----------------------------
# INITIALISATION
# ----------------------------
print("🚀 Démarrage de l'API...")

# Récupérer le token au démarrage
HF_TOKEN, USE_LLAMA = get_hf_token_on_startup()

# Choix du modèle
if USE_LLAMA:
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    print(f"\n🎯 Modèle API: {MODEL_NAME}")
else:
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\n🔓 Modèle open-source API: {MODEL_NAME}")

app = FastAPI(
    title="LLaMA API",
    description=f"REST API using {MODEL_NAME}",
    version="1.0.0"
)

# ----------------------------
# MODÈLE & TOKENIZER (chargement lazy)
# ----------------------------
model = None
tokenizer = None

def load_model():
    """Charge le modèle de manière lazy"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        print(f"🔄 Chargement du modèle {MODEL_NAME}...")
        
        try:
            # Configuration de quantisation
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # Tokenizer
            if USE_LLAMA and HF_TOKEN:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
            else:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            tokenizer.pad_token = tokenizer.eos_token
            
            # Modèle
            if USE_LLAMA and HF_TOKEN:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=HF_TOKEN
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            print(f"✅ Modèle {MODEL_NAME} chargé avec succès!")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            print("Utilisation d'un modèle alternatif...")
            # Fallback simple
            MODEL_NAME_FALLBACK = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FALLBACK)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME_FALLBACK,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"✅ Modèle de secours {MODEL_NAME_FALLBACK} chargé!")
    
    return model, tokenizer

# ----------------------------
# SCHÉMAS PYDANTIC (inchangés)
# ----------------------------
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text for generation")
    max_tokens: int = Field(150, ge=1, le=1000)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    json_schema: Optional[Dict[str, Any]] = Field(None)

class FactCheckRequest(BaseModel):
    statement: str = Field(..., description="Statement to fact-check")
    context: Optional[str] = Field(None)

class SQLGenerationRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    schema_info: str = Field(..., description="Database schema information")

# ----------------------------
# FONCTIONS UTILITAIRES
# ----------------------------
def format_prompt(user_message: str, system_message: str = None) -> str:
    """Formate un prompt selon le modèle"""
    if "llama" in MODEL_NAME.lower() and system_message:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}

<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    elif "llama" in MODEL_NAME.lower():
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        if system_message:
            return f"System: {system_message}\n\nUser: {user_message}\n\nAssistant: "
        else:
            return f"User: {user_message}\n\nAssistant: "

def extract_json_from_text(text: str) -> Dict:
    """Extrait et valide JSON depuis le texte généré"""
    import re
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except:
            try:
                json_str = json_str.replace("'", '"').replace("True", "true").replace("False", "false")
                return json.loads(json_str)
            except:
                return {"raw_text": text, "error": "Invalid JSON"}
    else:
        return {"raw_text": text, "error": "No JSON found"}

# ----------------------------
# ENDPOINTS API
# ----------------------------
@app.get("/")
async def root():
    return {
        "message": "LLaMA API is running",
        "model": MODEL_NAME,
        "requires_token": USE_LLAMA,
        "endpoints": ["/generate", "/generate_json", "/fact_check", "/generate_sql", "/health"]
    }

@app.get("/health")
async def health_check():
    try:
        load_model()
        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "requires_token": USE_LLAMA,
            "model_loaded": model is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    try:
        model_obj, tokenizer_obj = load_model()
        
        prompt = format_prompt(request.prompt)
        inputs = tokenizer_obj(prompt, return_tensors="pt").to(model_obj.device)
        
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
        
        generated_text = tokenizer_obj.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire la réponse
        if "assistant" in generated_text:
            response = generated_text.split("assistant")[-1].strip()
        elif "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        else:
            response = generated_text
        
        return {
            "model": MODEL_NAME,
            "