from pydantic import BaseModel


class HealthCheck(BaseModel):
    healthy: bool


class QuestionRequest(BaseModel):
    q: str


class QuestionResponse(BaseModel):
    answer: str


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
