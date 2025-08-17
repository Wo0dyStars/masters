import argparse
from pathlib import Path
from dotenv import load_dotenv
from utils import verify_store, load_existing_store

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Verify and inspect FAISS vector store")
    parser.add_argument("--path", type=Path, default="vector_store/stores/md_medium_recursive", help="Path to vector store (default: vector_store/stores/md_medium_recursive)")
    parser.add_argument("--query", default="test query", help="Test query for verification (default: 'test query')")
    parser.add_argument("--search", action="store_true", help="Perform sample search")
    
    args = parser.parse_args()
    results = verify_store(args.path, args.query)
    
    if results['error']:
        print(f"  - Error: {results['error']}")
    
    if args.search and results['searchable']:
        vectorstore = load_existing_store(args.path)
        if vectorstore:
            docs = vectorstore.similarity_search(args.query, k=3)
            
            for i, doc in enumerate(docs, 1):
                print(f"\n📄 Result {i}:")
                print(f"  Source: {doc.metadata.get('source_file', 'Unknown')}")
                print(f"  Content: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()