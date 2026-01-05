from llama_cpp import Llama


def init_model():
    print("Loading model...")
    llm = Llama.from_pretrained(
        repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF", filename="*q8_0.gguf", verbose=False
    )
    print("Model loaded.")
    return llm
