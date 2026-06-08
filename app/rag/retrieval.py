from app.embeddings.generator import EmbeddingGenerator
from typing import List, Dict, Any

class Retriever:
    """
    Mecanismo de busca (Retrieval) para o pipeline RAG, responsável por
    vetorizar a consulta do usuário, calcular a similaridade com os chunks
    armazenados e retornar os mais relevantes.
    """

    def __init__(self, generator: EmbeddingGenerator) -> None:
        """
        Inicializa o Retriever com uma instância treinada de EmbeddingGenerator.
        """
        self.generator = generator

    def search(
        self,
        query: str,
        embedded_chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Realiza a busca pelos chunks mais semelhantes à consulta do usuário.
        
        Etapas:
        1. Gera o embedding da consulta do usuário (query) através de transform().
        2. Para cada chunk no banco de chunks embutidos, extrai o embedding do chunk.
        3. Calcula a similaridade de cosseno entre o embedding da consulta e o embedding do chunk.
        4. Monta o dicionário resultante com os metadados originais e a pontuação (score) calculada.
        5. Ordena os resultados decrescentemente pelo score.
        6. Retorna os top_k chunks mais relevantes.
        """
        if not embedded_chunks:
            return []

        # 1. Gera embedding da pergunta
        query_embedding = self.generator.transform(query)

        results = []
        
        # 2 e 3. Calcula similaridade com todos os chunks
        for chunk in embedded_chunks:
            chunk_embedding = chunk.get("embedding", [])
            
            # Garante que haja um embedding no chunk
            if not chunk_embedding:
                similarity = 0.0
            else:
                similarity = self.generator.cosine_similarity(query_embedding, chunk_embedding)
            
            # 4. Formata o resultado do chunk
            results.append({
                "chunk_id": chunk.get("chunk_id"),
                "page": chunk.get("page"),
                "source": chunk.get("source"),
                "content": chunk.get("content"),
                "score": float(similarity)
            })

        # 5. Ordena por similaridade de cosseno decrescente
        results.sort(key=lambda x: x["score"], reverse=True)

        # 6. Retorna os top_k mais relevantes
        return results[:top_k]
