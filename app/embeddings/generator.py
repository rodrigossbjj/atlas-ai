import math
import re
from typing import List, Dict, Any

class EmbeddingGenerator:
    """
    Gerador de Embeddings educacional que implementa a representação vetorial TF-IDF
    e o cálculo de similaridade de cosseno do zero, utilizando apenas a biblioteca
    padrão do Python.
    """

    def __init__(self) -> None:
        self.vocabulary: List[str] = []
        self.word_to_idx: Dict[str, int] = {}
        self.idf_dict: Dict[str, float] = {}
        self.num_documents: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokeniza o texto de entrada: converte para minúsculas e extrai palavras alfanuméricas.
        Dessa forma, pontuações são removidas e caracteres especiais/acentuados são mantidos.

        Exemplo: "O RAG é incrível!" -> ["o", "rag", "é", "incrível"]
        """
        if not text or not isinstance(text, str):
            return []
        # O padrão \w+ captura sequências de caracteres alfanuméricos (letras, números, underscore).
        # Em Python 3, por padrão, o regex suporta Unicode, incluindo letras acentuadas.
        return re.findall(r'\w+', text.lower())

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Ajusta (treina) o modelo TF-IDF com base no corpus de chunks fornecido.
        
        Etapas:
        1. Tokeniza o conteúdo de todos os chunks.
        2. Constrói um vocabulário de termos únicos ordenados.
        3. Calcula a Document Frequency (DF) de cada termo (quantos documentos contêm o termo).
        4. Calcula a Inverse Document Frequency (IDF) suavizada para cada termo.
        """
        self.num_documents = len(chunks)
        if self.num_documents == 0:
            self.vocabulary = []
            self.word_to_idx = {}
            self.idf_dict = {}
            return

        # 1. Extração e Tokenização dos textos
        tokenized_docs = [self._tokenize(chunk.get("content", "")) for chunk in chunks]

        # 2. Construção do vocabulário único
        vocab_set = set()
        for doc in tokenized_docs:
            vocab_set.update(doc)
        
        self.vocabulary = sorted(list(vocab_set))
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}

        # 3. Cálculo de Document Frequency (DF)
        df_counts = {word: 0 for word in self.vocabulary}
        for doc in tokenized_docs:
            unique_words_in_doc = set(doc)
            for word in unique_words_in_doc:
                df_counts[word] += 1

        # 4. Cálculo de Inverse Document Frequency (IDF) suavizado
        # Fórmula padrão do scikit-learn: IDF(t) = ln((1 + N) / (1 + DF(t))) + 1
        # Isso previne que termos com DF=0 no corpus (se houvesse) ou termos que aparecem em todos
        # os documentos fiquem com IDF nulo ou negativo.
        self.idf_dict = {}
        for word in self.vocabulary:
            df = df_counts[word]
            self.idf_dict[word] = math.log((1 + self.num_documents) / (1 + df)) + 1

    def transform(self, text: str) -> List[float]:
        """
        Converte um texto em um vetor numérico TF-IDF com base no vocabulário aprendido no fit.
        
        Etapas:
        1. Tokeniza o texto de entrada.
        2. Calcula a frequência bruta dos termos do vocabulário no texto de entrada.
        3. Calcula o Term Frequency (TF) normalizado:
           TF(t, d) = (frequência de t em d) / (total de termos em d)
        4. Multiplica o TF de cada termo pelo IDF correspondente armazenado na classe.
        
        Retorna:
            List[float]: Vetor TF-IDF resultante com o mesmo comprimento que o vocabulário.
        """
        if not self.vocabulary:
            return []

        tokens = self._tokenize(text)
        total_tokens = len(tokens)

        # Contagem de frequência bruta dos termos na entrada
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Criação do vetor zerado
        vector = [0.0] * len(self.vocabulary)

        # Tratamento para textos vazios ou sem tokens válidos
        if total_tokens == 0:
            return vector

        # Preenche as posições correspondentes do vocabulário
        for word, idx in self.word_to_idx.items():
            count = token_counts.get(word, 0)
            # Frequência relativa do termo no documento
            tf = count / total_tokens
            # IDF previamente computado
            idf = self.idf_dict.get(word, 0.0)
            # TF-IDF
            vector[idx] = tf * idf

        return vector

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
