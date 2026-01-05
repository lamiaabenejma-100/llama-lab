from fastapi import FastAPI, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager
from llama_cpp import Llama

from llm import init_model
from api_models import (
    QuestionRequest,
    QuestionResponse,
    HealthCheck,
    ChatRequest,
)
from fastapi.middleware.cors import CORSMiddleware


# App setup:
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = init_model()
    yield
    # NOTE: Clean up resources if needed when the app is shutting down


# Init App:
app = FastAPI(lifespan=lifespan)


# CORS:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependencies:
def get_llm():
    return app.state.llm


# Routes:
@app.get("/")
def read_root():
    return HealthCheck(healthy=True)


@app.post("/question")
def get_answer(data: QuestionRequest, llm: Llama = Depends(get_llm)):
    answer = llm(
        f"Q: {data.q} A:",  # Prompt
        max_tokens=32,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
        stop=[
            "Q:",
            "\n",
        ],  # Stop generating just before the model would generate a new question
    )
    return QuestionResponse(answer=answer["choices"][0]["text"])


@app.post("/chat")
def chat(data: ChatRequest, llm: Llama = Depends(get_llm)):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions.",
        },
        *[
            {"role": message.role, "content": message.content}
            for message in data.messages
        ],
    ]
    answer = llm.create_chat_completion(
        messages=messages,
    )
    return QuestionResponse(answer=answer["choices"][0]["message"]["content"])


# Run:
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
