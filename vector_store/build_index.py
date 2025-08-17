import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from config import VectorStoreConfig, ChunkingStrategy
from builder import VectorStoreBuilder

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector store from documents")
    
    parser.add_argument("--docs", type=Path, default="docs", help="Path to documents folder (default: docs)")
    parser.add_argument("--output", type=Path, default="vector_store/stores/md_medium_recursive", help="Output path for vector store (default: vector_store/stores/md_medium_recursive)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for text splitting (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for text splitting (default: 200)")
    parser.add_argument("--strategy", choices=["recursive", "character", "markdown"], default="recursive", help="Text splitting strategy")
    parser.add_argument("--no-duplicates", action="store_true", help="Remove duplicate chunks")
    parser.add_argument("--min-length", type=int, default=50, help="Minimum chunk length (default: 50)")
    parser.add_argument("--max-length", type=int, default=4000, help="Maximum chunk length (default: 4000)")
    parser.add_argument("--force", action="store_true", help="Force rebuild if index already exists")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing (default: 100)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = VectorStoreConfig(
        docs_folder=args.docs,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=ChunkingStrategy(args.strategy),
        remove_duplicates=args.no_duplicates,
        min_chunk_length=args.min_length,
        max_chunk_length=args.max_length,
        batch_size=args.batch_size
    )
    
    builder = VectorStoreBuilder(config)
    success = builder.build(force_rebuild=args.force)
    
    if success:
        print(f"\nVector store built successfully!")
        print(f"Location: {args.output}")
        print(f"Use this path in your RAG applications")
    else:
        print(f"\nVector store build failed!")
        exit(1)

if __name__ == "__main__":
    main()