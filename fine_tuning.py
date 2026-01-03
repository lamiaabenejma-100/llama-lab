# fine_tuning.py
"""
Fine-tuning LLaMA 3.2-1B with LoRA/QLoRA
Parameter-efficient fine-tuning for sentiment analysis
"""

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
import os
from dotenv import load_dotenv
import numpy as np

# ----------------------------
# CONFIGURATION
# ----------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HF_TOKEN:
    print("❌ Token Hugging Face non trouvé!")
    exit(1)

login(token=HF_TOKEN)

# Modèle LLaMA 3.2
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATASET_NAME = "imdb"
OUTPUT_DIR = "./llama3.2-finetuned-lora"

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
print("FINE-TUNING LLaMA 3.2-1B WITH LoRA")
print("=" * 60)

# ----------------------------
# 1. CHARGEMENT DU MODÈLE ET TOKENIZER
# ----------------------------
print(f"\n1. Chargement du modèle {MODEL_NAME}...")

# Configuration de quantisation 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.float16
)

try:
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Modèle avec quantisation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
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
train_dataset = dataset["train"].select(range(1000))
eval_dataset = dataset["test"].select(range(200))

print(f"   • Exemples d'entraînement: {len(train_dataset)}")
print(f"   • Exemples d'évaluation: {len(eval_dataset)}")

# Fonction de préparation des données
def preprocess_function(examples):
    # Créer des prompts pour la classification de sentiments
    prompts = []
    for text, label in zip(examples["text"], examples["label"]):
        sentiment = "positive" if label == 1 else "negative"
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Analyze the sentiment of this movie review:

"{text[:500]}"

Sentiment:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{sentiment}<|eot_id|>"""
        prompts.append(prompt)
    
    # Tokenizer
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )
    
    # Les labels sont les input_ids pour le language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Appliquer le préprocessing
print("   • Préprocessing des données...")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=32,
    remove_columns=train_dataset.column_names
)

tokenized_eval = eval_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=32,
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
print("\n7. Évaluation avant/après fine-tuning")

def test_sentiment(text, model, tokenizer):
    """Teste le modèle sur un texte donné"""
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Analyze the sentiment of this movie review:

"{text}"

Sentiment:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
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
    sentiment = response.split("assistant")[-1].strip().lower()
    
    return sentiment

# Textes de test
test_texts = [
    "This movie was absolutely fantastic! The acting was superb and the story was captivating.",
    "I was very disappointed with this film. The plot was weak and the characters were uninteresting.",
    "An average movie with some good moments but overall nothing special."
]

print("\n📊 Résultats des tests:")
for i, text in enumerate(test_texts, 1):
    print(f"\nTest {i}:")
    print(f"   Texte: {text[:80]}...")
    sentiment = test_sentiment(text, model, tokenizer)
    print(f"   Sentiment prédit: {sentiment}")

print("\n" + "=" * 60)
print("✅ FINE-TUNING TERMINÉ AVEC SUCCÈS")
print("=" * 60)