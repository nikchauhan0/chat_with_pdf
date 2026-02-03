import argparse
import textwrap

from rag_chatbot.rag import RAGChatbot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG chatbot over PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              python -m rag_chatbot.cli --pdfs docs/a.pdf docs/b.pdf
            """
        ),
    )
    parser.add_argument("--pdfs", nargs="+", required=True, help="PDF file paths")
    parser.add_argument("--top-k", type=int, default=4, help="chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=800, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=120, help="overlap between chunks")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    bot = RAGChatbot(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        model=args.model,
    )
    bot.index(args.pdfs)

    print("PDFs indexed. Ask a question (type 'exit' to quit).\n")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        answer, retrieved = bot.chat(query)
        print("\nAssistant:")
        print(answer)
        print("\nSources:")
        for chunk in retrieved:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "?")
            print(f"- {source} (page {page}) score={chunk.score:.3f}")
        print("")


if __name__ == "__main__":
    main()
