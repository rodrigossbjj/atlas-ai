import argparse
from app.pdf.reader import read_pdf
from app.rag.chunker import create_chunks
from app.embeddings.generator import EmbeddingGenerator
from app.rag.retrieval import Retriever


def build_index(pdf_path: str) -> tuple[list[dict], EmbeddingGenerator]:
    document = read_pdf(pdf_path)
    chunks = create_chunks(document)

    generator = EmbeddingGenerator()

    texts = [chunk["content"] for chunk in chunks]

    embeddings = generator.transform_many(texts)

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded = chunk.copy()
        embedded["embedding"] = embedding
        embedded_chunks.append(embedded)

    return embedded_chunks, generator


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atlas AI — Pipeline RAG com TF-IDF do zero.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py caminho/para/documento.pdf "Sua pergunta aqui"
  python main.py caminho/para/documento.pdf "Sua pergunta aqui" --top_k 5
        """
    )
    parser.add_argument("pdf", help="Caminho para o arquivo PDF a ser indexado.")
    parser.add_argument("query", help="Pergunta a ser buscada no documento.")
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Número de resultados a retornar (padrão: 3)."
    )

    args = parser.parse_args()

    print(f"\nIndexando: {args.pdf}")
    embedded_chunks, generator = build_index(args.pdf)
    print(f"\nChunks indexados: {len(embedded_chunks)}")

    print(f'\nBuscando: "{args.query}"')
    results = search(args.query, embedded_chunks, generator, top_k=args.top_k)

    print_results(results)


if __name__ == "__main__":
    main()