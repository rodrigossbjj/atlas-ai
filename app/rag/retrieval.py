from app.embeddings.generator import EmbeddingGenerator
from typing import List, Dict, Any
import math 

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


    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Calcula a similaridade de cosseno entre dois vetores numéricos de mesmo tamanho.
        
        Fórmula matemática:
            Similarity = (v1 . v2) / (||v1|| * ||v2||)
            
            Onde:
            - (v1 . v2) é o produto escalar (dot product).
            - ||v1|| e ||v2|| são as normas euclidianas (L2-norm) dos vetores.
            
        Retorna:
            float: Similaridade entre 0.0 (totalmente dissimilares) e 1.0 (idênticos).
        """
        if len(v1) != len(v2):
            raise ValueError("Os vetores devem possuir o mesmo tamanho para o cálculo de similaridade.")

        if not v1:
            return 0.0

        # Produto escalar (dot product)
        dot_product = sum(x * y for x, y in zip(v1, v2))

        # Norma L2 do vetor 1: sqrt(sum(x_i^2))
        norm_v1 = math.sqrt(sum(x * x for x in v1))

        # Norma L2 do vetor 2: sqrt(y_i^2))
        norm_v2 = math.sqrt(sum(y * y for y in v2))

        # Tratamento de vetores de magnitude zero para evitar divisão por zero
        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0

        similarity = dot_product / (norm_v1 * norm_v2)

        # Garante que o score fique no intervalo [0.0, 1.0] contra eventuais imprecisões numéricas de float
        return max(0.0, min(1.0, similarity))

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
                similarity = self.cosine_similarity(query_embedding, chunk_embedding)
            
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
