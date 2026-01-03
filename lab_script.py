# lab_script_fixed.py
"""
Lab Assignment 3 - Version corrigée
"""

# =============== FIX SSL ===============
import os
import ssl
import warnings

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
warnings.filterwarnings("ignore")
# =======================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# ----------------------------
# CONFIGURATION SÉCURISÉE
# ----------------------------
def get_token():
    """Récupère le token de manière sécurisée"""
    
    # Essayer d'abord .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token
    except:
        pass
    
    # Demander à l'utilisateur
    print("\n" + "=" * 60)
    print("🔐 TOKEN HUGGING FACE REQUIS")
    print("=" * 60)
    
    print("\nPour obtenir un token:")
    print("1. Allez sur: https://huggingface.co/settings/tokens")
    print("2. Créez un token 'read'")
    print("3. Copiez-le (il commence par 'hf_')")
    print("\n⚠️  ATTENTION: Ne partagez jamais votre token !")
    
    token = input("\nEntrez votre token (masqué): ")
    
    # Sauvegarder optionnellement
    save = input("\nSauvegarder dans .env? (o/n): ").lower()
    if save == 'o':
        with open(".env", "w") as f:
            f.write(f"HUGGING_FACE_HUB_TOKEN={token}")
        print("✅ Token sauvegardé (ajoutez .env à .gitignore !)")
    
    return token

# Récupérer le token
HF_TOKEN = get_token()

# Authentification
try:
    login(token=HF_TOKEN)
    print("✅ Authentification réussie")
    USE_LLAMA = True
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("Utilisation d'un modèle open-source...")
    USE_LLAMA = False

# ----------------------------
# CHARGEMENT DU MODÈLE
# ----------------------------
print("\n" + "=" * 60)
print("1. CHARGEMENT DU MODÈLE")
print("=" * 60)

# Choix du modèle
if USE_LLAMA:
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    print(f"🎯 LLaMA 3.2-1B")
else:
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"🔓 TinyLlama 1.1B")

# Vérifier si GPU est disponible
if torch.cuda.is_available():
    device = "cuda"
    print("✅ GPU disponible")
else:
    device = "cpu"
    print("⚠️  GPU non disponible - Utilisation du CPU")

print(f"\nChargement de {MODEL_NAME}...")

try:
    # Charger le tokenizer
    if USE_LLAMA:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Configurer le tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Charger le modèle
    if USE_LLAMA:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            token=HF_TOKEN
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"✅ Modèle chargé sur {device}")
    print(f"📊 Paramètres: {model.num_parameters():,}")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("\nChargement d'un modèle léger...")
    MODEL_NAME = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)
    print(f"✅ {MODEL_NAME} chargé")

# ----------------------------
# FORMATAGE DES PROMPTS
# ----------------------------
def format_prompt_llama(user_message):
    """Format correct pour LLaMA 3.2"""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""

def format_prompt_general(user_message):
    """Format pour autres modèles"""
    return f"### Instruction:\n{user_message}\n\n### Response:\n"

# ----------------------------
# TEST SIMPLE
# ----------------------------
print("\n" + "=" * 60)
print("2. TEST DE GÉNÉRATION")
print("=" * 60)

# Choisir le bon format
if "llama" in MODEL_NAME.lower():
    prompt_text = format_prompt_llama("Explain AI to a 10-year-old.")
else:
    prompt_text = format_prompt_general("Explain AI to a 10-year-old.")

print("Prompt formaté:")
print(prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text)

# Tokenisation
inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

# Génération avec paramètres optimisés
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3  # Évite la répétition
    )

# Décodage
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extraire la réponse (après le prompt)
if "llama" in MODEL_NAME.lower():
    # Pour LLaMA format
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        answer = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    else:
        answer = response
else:
    # Pour format général
    if "### Response:" in response:
        answer = response.split("### Response:")[-1].strip()
    else:
        answer = response

print("\n🤖 Réponse générée:")
print("-" * 40)
print(answer)
print("-" * 40)

# ----------------------------
# DIFFÉRENTES STRATÉGIES
# ----------------------------
print("\n" + "=" * 60)
print("3. STRATÉGIES DE DÉCODAGE")
print("=" * 60)

strategies = [
    ("Greedy", {"do_sample": False, "temperature": None}),
    ("Sampling temp=0.7", {"do_sample": True, "temperature": 0.7}),
    ("Top-k k=50", {"do_sample": True, "top_k": 50, "temperature": 0.7}),
]

test_prompt = "What is machine learning?" if "llama" in MODEL_NAME.lower() else "What is machine learning?"
if "llama" in MODEL_NAME.lower():
    test_prompt = format_prompt_llama(test_prompt)
else:
    test_prompt = format_prompt_general(test_prompt)

inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

for name, params in strategies:
    print(f"\n🔧 {name}:")
    print("-" * 30)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            **params,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "llama" in MODEL_NAME.lower():
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            answer = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            answer = response[-100:]
    else:
        if "### Response:" in response:
            answer = response.split("### Response:")[-1].strip()
        else:
            answer = response[-100:]
    
    print(answer[:200])

print("\n" + "=" * 60)
print("✅ LAB TERMINÉ AVEC SUCCÈS !")
print("=" * 60)