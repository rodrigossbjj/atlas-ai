import math
import pytest
from app.embeddings.generator import EmbeddingGenerator
from app.rag.retrieval import Retriever


def test_tokenizer() -> None:
    generator = EmbeddingGenerator()
    
    # Test text with punctuation and accents
    text = "O RAG é incrível, não é? Vamos testar!"
    tokens = generator._tokenize(text)
    
    # Expected: lowered, punctuation removed, accents preserved
    expected = ["o", "rag", "é", "incrível", "não", "é", "vamos", "testar"]
    assert tokens == expected

    # Test empty or none values
    assert generator._tokenize("") == []
    assert generator._tokenize(None) == []  # type: ignore


def test_fit_and_vocabulary() -> None:
    generator = EmbeddingGenerator()
    chunks = [
        {"content": "o gato de botas"},
        {"content": "o cachorro de botas"}
    ]
    
    generator.fit(chunks)
    
    # Vocabulary should be sorted alphabetically
    assert generator.vocabulary == ["botas", "cachorro", "de", "gato", "o"]
    assert generator.word_to_idx == {
        "botas": 0,
        "cachorro": 1,
        "de": 2,
        "gato": 3,
        "o": 4
    }
    
    # Check Document Frequency and IDF Calculations
    # N = 2
    # DF("botas") = 2 => IDF("botas") = ln((1+2)/(1+2)) + 1 = ln(1) + 1 = 1.0
    # DF("cachorro") = 1 => IDF("cachorro") = ln((1+2)/(1+1)) + 1 = ln(1.5) + 1 ~ 1.4054651
    
    assert math.isclose(generator.idf_dict["botas"], 1.0)
    assert math.isclose(generator.idf_dict["cachorro"], math.log(1.5) + 1.0)
    assert generator.num_documents == 2


def test_transform() -> None:
    generator = EmbeddingGenerator()
    chunks = [
        {"content": "o gato de botas"},
        {"content": "o cachorro de botas"}
    ]
    generator.fit(chunks)
    
    # Test transformation of a normal string
    # "o gato" has 2 tokens: "o", "gato"
    # TF("o") = 1/2 = 0.5, IDF("o") = 1.0 => TF-IDF = 0.5
    # TF("gato") = 1/2 = 0.5, IDF("gato") = ln(1.5) + 1.0 ~ 1.4054651 => TF-IDF = 0.7027325
    # TF for others = 0.0 => TF-IDF = 0.0
    vector = generator.transform("o gato")
    assert len(vector) == 5
    assert math.isclose(vector[4], 0.5)  # "o" is at index 4
    assert math.isclose(vector[3], 0.5 * (math.log(1.5) + 1.0))  # "gato" is at index 3
    assert vector[0] == 0.0  # "botas"
    assert vector[1] == 0.0  # "cachorro"
    assert vector[2] == 0.0  # "de"

    # Test transformation with words outside the vocabulary
    vector_unknown = generator.transform("papagaio invisível")
    assert vector_unknown == [0.0, 0.0, 0.0, 0.0, 0.0]

    # Test empty input
    assert generator.transform("") == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_cosine_similarity() -> None:
    generator = EmbeddingGenerator()
    
    # Test normal vectors
    v1 = [1.0, 2.0, 3.0]
    v2 = [1.0, 2.0, 3.0]
    # Identical vectors should have similarity of 1.0
    assert math.isclose(generator.cosine_similarity(v1, v2), 1.0)

    # Completely orthogonal vectors
    v3 = [1.0, 0.0]
    v4 = [0.0, 1.0]
    assert math.isclose(generator.cosine_similarity(v3, v4), 0.0)

    # Zero vector case
    v_zero = [0.0, 0.0, 0.0]
    assert generator.cosine_similarity(v1, v_zero) == 0.0

    # General calculation check
    # A = [1, 2], B = [2, 3]
    # dot = 1*2 + 2*3 = 8
    # normA = sqrt(1 + 4) = sqrt(5) ~ 2.236
    # normB = sqrt(4 + 9) = sqrt(13) ~ 3.605
    # sim = 8 / (sqrt(5) * sqrt(13)) = 8 / sqrt(65) ~ 0.9922778
    assert math.isclose(generator.cosine_similarity([1.0, 2.0], [2.0, 3.0]), 8.0 / math.sqrt(65.0))


def test_retriever_search() -> None:
    generator = EmbeddingGenerator()
    chunks = [
        {"chunk_id": 1, "page": 1, "source": "doc.pdf", "content": "o gato comeu o peixe"},
        {"chunk_id": 2, "page": 1, "source": "doc.pdf", "content": "o cachorro roeu o osso"},
        {"chunk_id": 3, "page": 2, "source": "doc.pdf", "content": "o gato de botas correu"}
    ]
    generator.fit(chunks)
    
    # Embed the chunks
    embedded_chunks = []
    for chunk in chunks:
        embedded_chunk = chunk.copy()
        embedded_chunk["embedding"] = generator.transform(chunk["content"])
        embedded_chunks.append(embedded_chunk)
        
    retriever = Retriever(generator)
    
    # Search for "gato"
    # Should rank chunks 3 and 1 higher than chunk 2
    results = retriever.search("gato", embedded_chunks, top_k=2)
    
    assert len(results) == 2
    # First place should be either 1 or 3 (containing gato) and not 2
    assert results[0]["chunk_id"] in [1, 3]
    assert results[1]["chunk_id"] in [1, 3]
    assert results[0]["score"] > 0.0
    
    # Verify that fields are returned as expected
    for res in results:
        assert "chunk_id" in res
        assert "page" in res
        assert "source" in res
        assert "content" in res
        assert "score" in res
        assert "embedding" not in res  # Score should replace embedding in results
