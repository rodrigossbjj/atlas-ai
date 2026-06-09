from ollama import Client
from typing import List, Dict, Any

class EmbeddingGenerator:
    def __init__(self):
        self.client = Client(host="http://localhost:11434")

    def transform(self, text: str) -> list[float]:
        response = self.client.embed(
            model="nomic-embed-text",
            input=text
        )

        return response["embeddings"][0]
    
    def transform_many(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(
            model="nomic-embed-text",
            input=texts
        )

        return response["embeddings"]
