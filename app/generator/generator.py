from ollama import Client

class Generator:
    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434") -> None:
        self._client = Client(host=host)
        self._model = model

    def generate(self, prompt: str,) -> str:
        response = self._client.generate(
            model=self._model,
            prompt=prompt,
        )

        return response["response"]