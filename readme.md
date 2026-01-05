# Fastapi - Llama Service Implementation

Using FastAPI to create a llama service that can be use anywhere to talk with model.

### (the docker usage will be here soon)

## Installation

```bash
pip install -r requirements.txt
```

## Usage:

```bash
cd app
python main.py
```

When you run first time, the `init_modal` function will download Llama model from huggingface so it will take some time to download the model.

# Example for asking simple question:

```bash
curl -X POST \
  'http://localhost:8000/question' \
  -H 'Content-Type: application/json' \
  -d '{"q": "What is the capital of France?"}'
```

### Response:

```json
{ "answer": " The capital of France is Paris." }
```

# Using Chat Completion (with message history)

There is two role for messages, `user` and `system`. The user role is for the messages that user sends and the system role is for the messages that llama model sends.

If you send them in order, the model will understand the context and give you better answers.

### Example Usage:

```bash
curl -X POST \
  'http://localhost:8000/chat' \
  -H 'Content-Type: application/json' \
  -d '{
	"messages": [
      {
        "role": "user",
        "content": "What is capital of Turkey ?"
      },
      {
        "role": "system",
        "content": "The capital of Turkey is Ankara"
      },
      {
        "role": "user",
        "content": "How about Spain ?"
      }
    ]
  }'
```

### Response:

```json
{
  "answer": "The capital of Spain is Madrid."
}
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
