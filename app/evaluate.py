import requests

API_URL = "http://localhost:8000/chat"

# 1️⃣ Liste de conversations à tester
test_chats = [
    {
        "messages": [
            {"role": "user", "content": "What is the capital of Turkey ?"},
            {"role": "system", "content": "The capital of Turkey is Ankara"},
            {"role": "user", "content": "How about Spain ?"}
        ],
        "expected_answer": "Madrid"
    },
    {
        "messages": [
            {"role": "user", "content": "What is the capital of Morocco ?"},
            {"role": "system", "content": "The capital of Morocco is Rabat"},
            {"role": "user", "content": "How about Italy ?"}
        ],
        "expected_answer": "Rome"
    }
]

# 2️⃣ Fonction d’évaluation
def evaluate_chat():
    total = len(test_chats)
    correct = 0

    for chat in test_chats:
        response = requests.post(API_URL, json=chat).json()
        answer = response.get("answer", "").strip()

        if chat["expected_answer"].lower() in answer.lower():
            print(f"[✅] Réponse correcte: {answer}")
            correct += 1
        else:
            print(f"[❌] Mauvaise réponse: {answer} (attendu: {chat['expected_answer']})")

    print(f"\nScore final: {correct}/{total} ({(correct/total)*100:.1f}%)")

if __name__ == "__main__":
    evaluate_chat()
