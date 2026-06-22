from app.pdf.reader import read_pdf
from app.rag.chunker import create_chunks
from app.embeddings.generator import EmbeddingGenerator
from app.rag.retrieval import Retriever
from app.repositories import Repository
from app.generator import Generator
from app.prompts import PromptBuilder

def build_index(pdf_path: str) -> tuple[list[dict], EmbeddingGenerator]:
    document = read_pdf(pdf_path)
    chunks = create_chunks(document)

    embedding_generator = EmbeddingGenerator()
    embedding_generator.fit(chunks)

    texts = [chunk["content"] for chunk in chunks]

    embeddings = embedding_generator.transform_many(texts)

    embedded_chunks = []
    repository = Repository()

    for chunk, embedding in zip(chunks, embeddings):
        embedded = chunk.copy()
        embedded["embedding"] = embedding
        embedded_chunks.append(embedded)

        repository.save_chunk(
            chunk_id=embedded["chunk_id"],
            page=embedded["page"],
            source=embedded["source"],
            content=embedded["content"],
            embedding=embedding
        )

    return embedded_chunks, embedding_generator


def search(query: str, embedded_chunks: list[dict], generator: EmbeddingGenerator, top_k: int) -> list[dict]:
    """Executa a busca de similaridade e retorna os top_k resultados."""
    retriever = Retriever(generator)
    return retriever.search(query, embedded_chunks, top_k=top_k)


def print_results(results: list[dict]) -> None:
    """Exibe os resultados da busca no terminal."""
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  TOP {len(results)} RESULTADOS")
    print(separator)

    for i, res in enumerate(results, 1):
        print(f"\n[{i}º] Score: {res['score']:.4f} | {res['source']} — Pág. {res['page']} | Chunk #{res['chunk_id']}")
        print(f"     {res['content'][:300]}...")

    print(f"\n{separator}\n")


def atlas_service(path: str, query: str, top_k: int = 3) -> str:

    print(f"\nIndexando: {path}")
    repository = Repository()

    if repository.count() == 0:
        print("Banco vazio. Indexando PDF...")
        embedded_chunks, generator = build_index(path)
    else:
        print("Carregando chunks do PostgreSQL...")
        embedded_chunks = repository.get_all_chunks()
        generator = EmbeddingGenerator()
        generator.fit(embedded_chunks)
    print(f"\nChunks indexados: {len(embedded_chunks)}")

    print(f'\nBuscando: "{query}"')
    results = search(query, embedded_chunks, generator, top_k)
 
    context = "\n\n".join(result["content"] for result in results)

    promp_builder = PromptBuilder()
    prompt = promp_builder.build(question=query, context=context)

    llm_generator = Generator()
    resp = llm_generator.generate(prompt=prompt)
    
    return resp
