class PromptBuilder:
    def build(self,question: str,context: str) -> str:
      return f"""
        Você é um assistente que responde perguntas usando apenas o contexto fornecido.

        Se a resposta não estiver presente no contexto, informe que não possui informações suficientes para responder.
        A resposta deve ser dada somente no idioma da pergunta.

        Contexto:
        {context}

        Pergunta:
        {question}

        Resposta:
        """.strip()