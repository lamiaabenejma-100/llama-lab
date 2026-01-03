# lab_script.py
"""
Lab Assignment 3 - Practical Introduction to LLaMA Models
Implementation using meta-llama/Llama-3.2-1B from Hugging Face
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os
from dotenv import load_dotenv

# ----------------------------
# 0. AUTHENTIFICATION HUGGING FACE
# ----------------------------
load_dotenv()  # Charger le token depuis .env

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    print("⚠️  ERREUR: Token Hugging Face non trouvé!")
    print("Veuillez créer un fichier .env avec HUGGING_FACE_HUB_TOKEN=votre_token")
    print("Ou exécutez: export HUGGING_FACE_HUB_TOKEN=votre_token")
    exit(1)

try:
    login(token=HF_TOKEN)
    print("✅ Authentification Hugging Face réussie")
except Exception as e:
    print(f"❌ Erreur d'authentification: {e}")
    exit(1)

# ----------------------------
# 1. SETUP & MODEL OVERVIEW - LLaMA 3.2 1B
# ----------------------------
print("\n" + "=" * 60)
print("1. SETUP & MODEL OVERVIEW - LLaMA 3.2 1B")
print("=" * 60)

MODEL_NAME = "meta-llama/Llama-3.2-1B"
print(f"Chargement du modèle: {MODEL_NAME}")

# Configuration de quantisation 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

try:
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Charger le modèle avec quantisation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    
    print("✅ Modèle LLaMA 3.2-1B chargé avec succès!")
    print(f"\n📊 Informations du modèle:")
    print(f"   • Architecture: {model.config.model_type}")
    print(f"   • Taille du vocabulaire: {model.config.vocab_size:,}")
    print(f"   • Longueur de contexte: {model.config.max_position_embeddings}")
    print(f"   • Paramètres totaux: {model.num_parameters():,}")
    print(f"   • Device: {model.device}")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    print("\nUtilisation d'un modèle de repli...")
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

# ----------------------------
# 2. BASIC INFERENCE & PROMPTING
# ----------------------------
print("\n" + "=" * 60)
print("2. BASIC INFERENCE & PROMPTING - Stratégies de décodage")
print("=" * 60)

# Fonction pour formater le prompt pour LLaMA 3.2
def format_llama_prompt(user_message):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

prompt_text = "Explain transformers (the AI model) to a 12-year-old."
prompt = format_llama_prompt(prompt_text)

print("Prompt original:")
print(f"  '{prompt_text}'")
print("\nPrompt formaté pour LLaMA 3.2:")
print(f"  '{prompt[:100]}...'")

# Encodage
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

# Tests des différentes stratégies de décodage
strategies = {
    "Greedy Decoding": {
        "do_sample": False,
        "max_new_tokens": 150
    },
    "Sampling (temp=0.7)": {
        "do_sample": True,
        "temperature": 0.7,
        "max_new_tokens": 150
    },
    "Top-k Sampling (k=50)": {
        "do_sample": True,
        "top_k": 50,
        "max_new_tokens": 150
    },
    "Top-p Sampling (p=0.9)": {
        "do_sample": True,
        "top_p": 0.9,
        "max_new_tokens": 150
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
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraire seulement la réponse de l'assistant
    response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    print(response[:300] + "..." if len(response) > 300 else response)

# ----------------------------
# 3. PROMPT ENGINEERING
# ----------------------------
print("\n" + "=" * 60)
print("3. PROMPT ENGINEERING - Techniques avancées")
print("=" * 60)

# a) Zero-shot
zero_shot_prompt = format_llama_prompt(
    "Is the following statement true or false? The Earth orbits the Sun."
)
print("\n🔍 Zero-shot Prompting:")
inputs = tokenizer(zero_shot_prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response.split("assistant")[-1].strip() if "assistant" in response else response[-200:])

# b) One-shot
one_shot_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Q: What is the capital of France?
A: Paris

Q: What is the capital of Germany?
A:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
print("\n🎯 One-shot Prompting:")
inputs = tokenizer(one_shot_prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True).split("A:")[-1].strip())

# c) Few-shot
few_shot_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Q: 2 + 2 = ?
A: 4
Q: 5 * 3 = ?
A: 15
Q: 10 / 2 = ?
A:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
print("\n📚 Few-shot Prompting (3 examples):")
inputs = tokenizer(few_shot_prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True).split("A:")[-1].strip())

# d) System-style role prompt
system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful and precise assistant specialized in explaining technical concepts simply.

<|start_header_id|>user<|end_header_id|>

What is machine learning?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
print("\n🤖 System-style Role Prompt:")
inputs = tokenizer(system_prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response.split("assistant")[-1].strip()[:300])

# e) Fact-checking avec sortie JSON
fact_prompt = format_llama_prompt(
    """Fact-check the following statement: "The Moon is made of cheese."
    Provide your answer in valid JSON format with these exact keys:
    - "statement": the original statement
    - "truthfulness": either "true", "false", or "partially true"
    - "explanation": brief explanation
    - "confidence": "high", "medium", or "low"
    
    JSON output:"""
)
print("\n✅ Fact-checking with Structured JSON Output:")
inputs = tokenizer(fact_prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.3)
response = tokenizer.decode(output[0], skip_special_tokens=True)
json_part = response.split("JSON output:")[-1].strip()
print(json_part[:400])

print("\n" + "=" * 60)
print("✅ LAB SCRIPT TERMINÉ AVEC SUCCÈS")
print("=" * 60)