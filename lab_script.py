# lab_script.py
"""
Lab Assignment 3 - Practical Introduction to LLaMA Models
Version avec demande interactive du token si nécessaire
"""

# =============== AJOUTEZ CES LIGNES POUR FIX SSL ===============
import os
import ssl
import warnings
import sys

# Fix pour les problèmes de certificat SSL
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
warnings.filterwarnings("ignore")
# ================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# ----------------------------
# 0. FONCTION POUR DEMANDER LE TOKEN INTERACTIVEMENT
# ----------------------------
def get_hf_token_interactive():
    """Demande le token à l'utilisateur de manière interactive"""
    
    print("\n" + "=" * 60)
    print("🔐 AUTHENTIFICATION HUGGING FACE")
    print("=" * 60)
    
    print("\nPour utiliser LLaMA 3.2, vous avez besoin d'un token Hugging Face.")
    print("\nSi vous n'avez pas de token, vous pouvez:")
    print("1. Utiliser un modèle open-source (appuyez sur Entrée)")
    print("2. Ou obtenir un token sur: https://huggingface.co/settings/tokens")
    
    choice = input("\nVoulez-vous entrer un token? (o/N): ").strip().lower()
    
    if choice == 'o' or choice == 'oui':
        print("\n" + "-" * 40)
        print("INSTRUCTIONS:")
        print("1. Allez sur: https://huggingface.co/settings/tokens")
        print("2. Créez un nouveau token (niveau 'read')")
        print("3. Copiez le token (commence par 'hf_')")
        print("-" * 40)
        
        token = input("\nEntrez votre token Hugging Face: ").strip()
        
        # Vérification basique
        if token and token.startswith('hf_'):
            # Option: sauvegarder dans .env
            save = input("\nVoulez-vous sauvegarder dans .env pour la prochaine fois? (o/N): ").strip().lower()
            if save == 'o' or save == 'oui':
                with open(".env", "w") as f:
                    f.write(f"HUGGING_FACE_HUB_TOKEN={token}")
                print("✅ Token sauvegardé dans .env")
            
            return token
        else:
            print("⚠️  Token invalide ou vide. Utilisation d'un modèle open-source.")
            return None
    else:
        print("✅ Utilisation d'un modèle open-source (pas de token requis)")
        return None

# ----------------------------
# 1. RÉCUPÉRATION DU TOKEN
# ----------------------------
HF_TOKEN = None
USE_LLAMA = False

# Essayer d'abord les sources automatiques
try:
    from dotenv import load_dotenv
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if HF_TOKEN:
        print("✅ Token trouvé dans .env")
        USE_LLAMA = True
except:
    pass

# Si pas de token, demander interactivement
if not HF_TOKEN:
    HF_TOKEN = get_hf_token_interactive()
    if HF_TOKEN:
        USE_LLAMA = True

# Authentification si token disponible
if USE_LLAMA and HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ Authentification Hugging Face réussie")
    except Exception as e:
        print(f"⚠️  Erreur d'authentification: {e}")
        print("Utilisation d'un modèle open-source à la place...")
        USE_LLAMA = False

# ----------------------------
# 2. SETUP & MODEL OVERVIEW
# ----------------------------
print("\n" + "=" * 60)
print("1. SETUP & MODEL OVERVIEW")
print("=" * 60)

# Choix du modèle selon l'accès
if USE_LLAMA:
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    print(f"🎯 Modèle LLaMA sélectionné: {MODEL_NAME}")
else:
    # Modèle open-source alternatif
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"🔓 Modèle open-source sélectionné: {MODEL_NAME}")

print(f"\nChargement du modèle: {MODEL_NAME}")

try:
    # Configuration de quantisation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Charger le modèle avec quantisation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("✅ Modèle chargé avec succès!")
    print(f"\n📊 Informations du modèle:")
    print(f"   • Architecture: {model.config.model_type}")
    print(f"   • Taille du vocabulaire: {model.config.vocab_size:,}")
    print(f"   • Device: {model.device}")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    print("\nTentative avec un modèle plus léger...")
    
    # Fallback ultra-léger
    MODEL_NAME = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ Modèle de secours {MODEL_NAME} chargé!")

# ----------------------------
# 3. BASIC INFERENCE & PROMPTING
# ----------------------------
print("\n" + "=" * 60)
print("2. BASIC INFERENCE & PROMPTING - Stratégies de décodage")
print("=" * 60)

# Fonction pour formater le prompt selon le modèle
def format_prompt(user_message):
    if "llama" in MODEL_NAME.lower():
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    elif "phi" in MODEL_NAME.lower():
        return f"Instruct: {user_message}\nOutput:"
    else:
        return f"<|user|>\n{user_message}\n<|assistant|>\n"

prompt_text = "Explain transformers (the AI model) to a 12-year-old."
prompt = format_prompt(prompt_text)

print("Prompt original:")
print(f"  '{prompt_text}'")
print("\nPrompt formaté:")
print(f"  '{prompt[:80]}...'")

# Encodage
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Tests des différentes stratégies de décodage
strategies = {
    "Greedy Decoding": {
        "do_sample": False,
        "max_new_tokens": 100
    },
    "Sampling (temp=0.7)": {
        "do_sample": True,
        "temperature": 0.7,
        "max_new_tokens": 100
    },
    "Top-k Sampling (k=50)": {
        "do_sample": True,
        "top_k": 50,
        "max_new_tokens": 100
    }
}

for strategy_name, params in strategies.items():
    print(f"\n{'─' * 40}")
    print(f"🎯 {strategy_name}")
    print(f"{'─' * 40}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **params,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Décodage et affichage
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraire seulement la réponse après le prompt
    response_text = response[len(prompt):].strip()
    print(response_text[:250] + "..." if len(response_text) > 250 else response_text)

# ----------------------------
# 4. PROMPT ENGINEERING
# ----------------------------
print("\n" + "=" * 60)
print("3. PROMPT ENGINEERING - Techniques avancées")
print("=" * 60)

# a) Zero-shot
zero_shot_prompt = format_prompt("Is the following statement true or false? The Earth orbits the Sun.")
print("\n🔍 Zero-shot Prompting:")
inputs = tokenizer(zero_shot_prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response.split("assistant")[-1].strip() if "assistant" in response else response[-150:])

# b) One-shot
one_shot_text = """Q: What is the capital of France?
A: Paris

Q: What is the capital of Germany?
A:"""
one_shot_prompt = format_prompt(one_shot_text)
print("\n🎯 One-shot Prompting:")
inputs = tokenizer(one_shot_prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
response = tokenizer.decode(output[0], skip_special_tokens=True)
answer = response.split("A:")[-1].strip().split("\n")[0]
print(f"Réponse: {answer}")

# c) Fact-checking avec sortie structurée
fact_text = """Fact-check the following statement: "The Moon is made of cheese."
Provide your answer in this format:
- Statement: [original statement]
- Truthfulness: [true/false/partially true]
- Explanation: [brief explanation]"""

fact_prompt = format_prompt(fact_text)
print("\n✅ Fact-checking with Structured Output:")
inputs = tokenizer(fact_prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.3)
response = tokenizer.decode(output[0], skip_special_tokens=True)
fact_response = response.split("assistant")[-1].strip() if "assistant" in response else response[-300:]
print(fact_response[:300])

print("\n" + "=" * 60)
print("✅ LAB SCRIPT TERMINÉ AVEC SUCCÈS!")
print("=" * 60)