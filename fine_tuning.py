# fine_tuning.py
"""
Fine-tuning avec LoRA/QLoRA
Version avec demande interactive du token si nécessaire
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

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from huggingface_hub import login
import numpy as np

# ----------------------------
# FONCTION POUR DEMANDER LE TOKEN
# ----------------------------
def get_hf_token_interactive():
    """Demande le token à l'utilisateur de manière interactive"""
    
    print("\n" + "=" * 60)
    print("🔐 AUTHENTIFICATION POUR FINE-TUNING")
    print("=" * 60)
    
    print("\nPour fine-tuner LLaMA 3.2, un token Hugging Face est requis.")
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
        print("✅ Utilisation d'un modèle open-source pour le fine-tuning")
        return None

# ----------------------------
# CONFIGURATION
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

# Choix du modèle
if USE_LLAMA:
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    print(f"\n🎯 Modèle pour fine-tuning: {MODEL_NAME}")
else:
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\n🔓 Modèle open-source pour fine-tuning: {MODEL_NAME}")

DATASET_NAME = "imdb"
OUTPUT_DIR = "./llama-finetuned-lora"

# Paramètres LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Paramètres d'entraînement
EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4

print("=" * 60)
print("FINE-TUNING AVEC LoRA/QLoRA")
print("=" * 60)

# ----------------------------
# 1. CHARGEMENT DU MODÈLE ET TOKENIZER
# ----------------------------
print(f"\n1. Chargement du modèle {MODEL_NAME}...")

# Configuration de quantisation
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.float16
)

try:
    # Tokenizer
    if USE_LLAMA and HF_TOKEN:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Modèle avec quantisation
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
    
    # Préparer le modèle pour l'entraînement k-bit
    model = prepare_model_for_kbit_training(model)
    
    print("✅ Modèle chargé avec succès!")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("Utilisation d'un modèle alternatif...")
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

# ----------------------------
# 2. CONFIGURATION LoRA
# ----------------------------
print("\n2. Configuration LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    modules_to_save=["lm_head", "embed_tokens"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ----------------------------
# 3. PRÉPARATION DES DONNÉES
# ----------------------------
print(f"\n3. Chargement du dataset {DATASET_NAME}...")

# Charger le dataset IMDB
dataset = load_dataset(DATASET_NAME)

# Prendre un sous-ensemble pour l'entraînement rapide
train_dataset = dataset["train"].select(range(500))  # Réduit pour plus de rapidité
eval_dataset = dataset["test"].select(range(100))

print(f"   • Exemples d'entraînement: {len(train_dataset)}")
print(f"   • Exemples d'évaluation: {len(eval_dataset)}")

# Fonction de préparation des données
def preprocess_function(examples):
    # Format simplifié pour accélérer
    prompts = []
    for text, label in zip(examples["text"], examples["label"]):
        sentiment = "positive" if label == 1 else "negative"
        if "llama" in MODEL_NAME.lower():
            prompt = f"""Review: {text[:300]}\nSentiment: {sentiment}"""
        else:
            prompt = f"""Analyze sentiment: {text[:300]}\nSentiment: {sentiment}"""
        prompts.append(prompt)
    
    # Tokenizer
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors=None
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Appliquer le préprocessing
print("   • Préprocessing des données...")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=16,
    remove_columns=train_dataset.column_names
)

tokenized_eval = eval_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=16,
    remove_columns=eval_dataset.column_names
)

print(f"   • Taille finale train: {len(tokenized_train)}")
print(f"   • Taille finale eval: {len(tokenized_eval)}")

# ----------------------------
# 4. CONFIGURATION DE L'ENTRAÎNEMENT
# ----------------------------
print("\n4. Configuration de l'entraînement...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=50,
    logging_steps=25,
    eval_steps=100,
    save_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=LEARNING_RATE,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ----------------------------
# 5. ENTRAÎNEMENT
# ----------------------------
print("\n5. Début de l'entraînement...")
print("   (Cela peut prendre quelques minutes)")

train_result = trainer.train()

# Sauvegarde
print(f"\n6. Sauvegarde du modèle dans {OUTPUT_DIR}...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Entraînement terminé!")
print(f"   • Loss finale: {train_result.training_loss:.4f}")

# ----------------------------
# 6. ÉVALUATION AVANT/APRÈS
# ----------------------------
print("\n7. Test du modèle fine-tuné")

def test_sentiment(text):
    """Teste le modèle sur un texte donné"""
    if "llama" in MODEL_NAME.lower():
        prompt = f"Review: {text}\nSentiment:"
    else:
        prompt = f"Analyze sentiment: {text}\nSentiment:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sentiment = response.split("Sentiment:")[-1].strip().lower()
    
    return sentiment

# Textes de test
test_texts = [
    "This movie was absolutely fantastic! The acting was superb.",
    "I was very disappointed with this film. The plot was weak.",
    "An average movie with some good moments."
]

print("\n📊 Résultats des tests:")
for i, text in enumerate(test_texts, 1):
    print(f"\nTest {i}:")
    print(f"   Texte: {text[:60]}...")
    sentiment = test_sentiment(text)
    print(f"   Sentiment prédit: {sentiment}")

print("\n" + "=" * 60)
print("✅ FINE-TUNING TERMINÉ AVEC SUCCÈS")
print("=" * 60)