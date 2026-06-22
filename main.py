import argparse
import app.service.atlas as atlas

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

    response = atlas.atlas_service(args.pdf, args.query, args.top_k)
    print(response)


if __name__ == "__main__":
    main()