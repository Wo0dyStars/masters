import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from builder import VectorStoreBuilder
from config import VectorStoreConfig, CHUNKING_CONFIGS, DocumentFormat

def build_single_store(config_name: str = None, config_index: int = 0):
    """Build a single vector store from chunking configuration"""
    
    if config_name:
        chunking_config = None
        for config in CHUNKING_CONFIGS:
            if config.name == config_name:
                chunking_config = config
                break
        if not chunking_config:
            print(f"Configuration '{config_name}' not found!")
            return False
    else:
        if config_index >= len(CHUNKING_CONFIGS):
            print(f"Index {config_index} out of range (max: {len(CHUNKING_CONFIGS)-1})")
            return False
        chunking_config = CHUNKING_CONFIGS[config_index]
    
    print(f"Building vector store: {chunking_config.name}")
    print(f"Description: {chunking_config.description}")
    print(f"Source: {chunking_config.folder_path}")
    print(f"Output: {chunking_config.output_path}")
    print(f"Strategy: {chunking_config.strategy.value}")
    print(f"Chunk size: {chunking_config.chunk_size} (overlap: {chunking_config.chunk_overlap})")
    
    vector_config = VectorStoreConfig(
        docs_folder=Path(chunking_config.folder_path),
        output_path=Path(chunking_config.output_path),
        chunk_size=chunking_config.chunk_size,
        chunk_overlap=chunking_config.chunk_overlap,
        chunking_strategy=chunking_config.strategy,
        
        supported_formats=[".pdf"] if chunking_config.format == DocumentFormat.PDF else [".md", ".txt"],
        
        remove_duplicates=True,
        min_chunk_length=50,
        max_chunk_length=4000,
        batch_size=100
    )
    
    if not vector_config.docs_folder.exists():
        print(f"Source folder not found: {vector_config.docs_folder}")
        return False
    
    if vector_config.output_path.exists():
        response = input(f"Output folder already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return False
    
    try:
        builder = VectorStoreBuilder(vector_config)
        success = builder.build(force_rebuild=True)
        
        if success:
            print(f"\nSuccessfully built vector store: {chunking_config.name}")
            print(f"Location: {chunking_config.output_path}")
            
            from utils import verify_store
            results = verify_store(Path(chunking_config.output_path))
            print(f"Verification: {results['document_count']} documents indexed")
            
            return True
        else:
            print(f"Failed to build vector store: {chunking_config.name}")
            return False
            
    except Exception as e:
        print(f"Error building vector store: {e}")
        return False

def list_configs():
    """List all available configurations"""
    print("Available Chunking Configurations:")
    print("=" * 50)
    for i, config in enumerate(CHUNKING_CONFIGS):
        print(f"{i:2d}. {config.name}")
        print(f"{config.description}")
        print(f"{config.folder_path}")
        print(f"{config.strategy.value} | Size: {config.chunk_size} | Overlap: {config.chunk_overlap}")
        print()

def main():
    """Main function with CLI interface"""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "list":
            list_configs()
            return
        elif arg.isdigit():
            index = int(arg)
            build_single_store(config_index=index)
        else:
            build_single_store(config_name=arg)
    else:
        print("Building first configuration (pdf_small_recursive)...")
        print("Usage: python build_store.py [index|name|list]")
        print()
        build_single_store(config_index=0)

if __name__ == "__main__":
    main()