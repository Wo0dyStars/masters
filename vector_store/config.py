from dataclasses import dataclass
from pathlib import Path
from typing import List
from enum import Enum

class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    CHARACTER = "character"
    MARKDOWN_AWARE = "markdown"

@dataclass
class VectorStoreConfig:
    """Configuration for vector store building"""
    
    docs_folder: Path = Path("docs")
    output_path: Path = Path("index")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    supported_formats: List[str] = None
    embedding_model: str = "text-embedding-ada-002"
    batch_size: int = 100
    distance_metric: str = "cosine"
    remove_duplicates: bool = True
    min_chunk_length: int = 50
    max_chunk_length: int = 4000
    include_source: bool = True
    include_page_numbers: bool = True
    include_file_metadata: bool = True
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".pdf", ".txt", ".md", ".docx"]
        
        self.docs_folder = Path(self.docs_folder)
        self.output_path = Path(self.output_path)

DEFAULT_CONFIG = VectorStoreConfig()

class DocumentFormat(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"

@dataclass
class ChunkingConfig:
    """Configuration for a single chunking experiment"""
    name: str
    format: DocumentFormat
    strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    description: str
    folder_path: str
    output_path: str

CHUNKING_CONFIGS = [
    
    ChunkingConfig(
        name="pdf_small_recursive",
        format=DocumentFormat.PDF,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=500,
        chunk_overlap=50,
        description="PDF with small recursive chunks - high precision, less context",
        folder_path="docs/PDF",
        output_path="vector_store/stores/pdf_small_recursive"
    ),
    
    ChunkingConfig(
        name="pdf_medium_recursive",
        format=DocumentFormat.PDF,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1000,
        chunk_overlap=200,
        description="PDF with medium recursive chunks - balanced approach (baseline)",
        folder_path="docs/PDF",
        output_path="vector_store/stores/pdf_medium_recursive"
    ),
    
    ChunkingConfig(
        name="pdf_large_recursive",
        format=DocumentFormat.PDF,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1500,
        chunk_overlap=300,
        description="PDF with large recursive chunks - more context, potential noise",
        folder_path="docs/PDF",
        output_path="vector_store/stores/pdf_large_recursive"
    ),
    
    ChunkingConfig(
        name="pdf_character_based",
        format=DocumentFormat.PDF,
        strategy=ChunkingStrategy.CHARACTER,
        chunk_size=1000,
        chunk_overlap=200,
        description="PDF with character-based splitting - simple boundary splitting",
        folder_path="docs/PDF",
        output_path="vector_store/stores/pdf_character_based"
    ),
    
    ChunkingConfig(
        name="pdf_high_overlap",
        format=DocumentFormat.PDF,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1000,
        chunk_overlap=400,
        description="PDF with high overlap - better context continuity but more redundancy",
        folder_path="docs/PDF",
        output_path="vector_store/stores/pdf_high_overlap"
    ),
    
    ChunkingConfig(
        name="md_small_recursive",
        format=DocumentFormat.MARKDOWN,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=500,
        chunk_overlap=50,
        description="Markdown with small recursive chunks - precise sections",
        folder_path="docs/Markdown",
        output_path="vector_store/stores/md_small_recursive"
    ),
    
    ChunkingConfig(
        name="md_medium_recursive", 
        format=DocumentFormat.MARKDOWN,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1000,
        chunk_overlap=200,
        description="Markdown with medium recursive chunks - balanced approach (baseline)",
        folder_path="docs/Markdown",
        output_path="vector_store/stores/md_medium_recursive"
    ),
    
    ChunkingConfig(
        name="md_large_recursive",
        format=DocumentFormat.MARKDOWN,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1500,
        chunk_overlap=300,
        description="Markdown with large recursive chunks - complete sections with context",
        folder_path="docs/Markdown",
        output_path="vector_store/stores/md_large_recursive"
    ),
    
    ChunkingConfig(
        name="md_structure_aware",
        format=DocumentFormat.MARKDOWN,
        strategy=ChunkingStrategy.MARKDOWN_AWARE,
        chunk_size=1000,
        chunk_overlap=200,
        description="Markdown with structure-aware splitting - respects headers and sections",
        folder_path="docs/Markdown",
        output_path="vector_store/stores/md_structure_aware"
    ),
    
    ChunkingConfig(
        name="md_high_overlap",
        format=DocumentFormat.MARKDOWN,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1000,
        chunk_overlap=400,
        description="Markdown with high overlap - maximum context preservation",
        folder_path="docs/Markdown",
        output_path="vector_store/stores/md_high_overlap"
    )
]