
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


