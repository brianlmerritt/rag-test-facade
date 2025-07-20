from typing import List, Dict, Any, Optional
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    SpacyTextSplitter
)
import tiktoken
import asyncio
import structlog
from sentence_transformers import SentenceTransformer
import spacy

logger = structlog.get_logger()

class ChunkingService:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedding_model = None
        self.spacy_model = None
        
    async def _load_embedding_model(self):
        """Lazy load embedding model for semantic chunking."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model
    
    async def _load_spacy_model(self):
        """Lazy load spacy model for sentence chunking."""
        if self.spacy_model is None:
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, falling back to NLTK")
                import nltk
                nltk.download('punkt', quiet=True)
                from nltk.tokenize import sent_tokenize
                self.spacy_model = sent_tokenize
        return self.spacy_model
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    async def create_chunks(
        self,
        text: str,
        strategy: str = "semantic",
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from text using specified strategy.
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy (semantic, fixed, sentence, paragraph)
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            metadata: Additional metadata to include
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if metadata is None:
            metadata = {}
        
        logger.info(
            "Creating chunks",
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            text_length=len(text)
        )
        
        chunks = []
        
        if strategy == "fixed":
            chunks = await self._fixed_chunking(text, chunk_size, overlap)
        elif strategy == "sentence":
            chunks = await self._sentence_chunking(text, chunk_size, overlap)
        elif strategy == "paragraph":
            chunks = await self._paragraph_chunking(text, chunk_size, overlap)
        elif strategy == "semantic":
            chunks = await self._semantic_chunking(text, chunk_size, overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk["metadata"] = {
                **metadata,
                "chunk_index": i,
                "chunk_strategy": strategy,
                "chunk_size_tokens": self._count_tokens(chunk["text"]),
                "total_chunks": len(chunks)
            }
        
        logger.info(f"Created {len(chunks)} chunks using {strategy} strategy")
        return chunks
    
    async def _fixed_chunking(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Fixed-size chunking with token-based splitting."""
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            encoding_name="cl100k_base"
        )
        
        text_chunks = splitter.split_text(text)
        return [{"text": chunk} for chunk in text_chunks]
    
    async def _sentence_chunking(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Sentence-aware chunking."""
        try:
            nlp = await self._load_spacy_model()
            if callable(nlp):  # NLTK fallback
                sentences = nlp(text)
            else:  # spaCy
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
        except Exception as e:
            logger.warning(f"Sentence segmentation failed: {e}, falling back to fixed chunking")
            return await self._fixed_chunking(text, chunk_size, overlap)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunks.append({"text": current_chunk.strip()})
                
                # Start new chunk with overlap
                if overlap > 0 and len(chunks) > 0:
                    # Take last few sentences for overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({"text": current_chunk.strip()})
        
        return chunks
    
    async def _paragraph_chunking(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Paragraph-aware chunking."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self._count_tokens(paragraph)
            
            # If paragraph is too large, split it further
            if paragraph_tokens > chunk_size:
                # If we have a current chunk, finalize it
                if current_chunk:
                    chunks.append({"text": current_chunk.strip()})
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large paragraph using recursive splitting
                sub_chunks = await self._fixed_chunking(paragraph, chunk_size, overlap)
                chunks.extend(sub_chunks)
                continue
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_tokens + paragraph_tokens > chunk_size and current_chunk:
                chunks.append({"text": current_chunk.strip()})
                
                # Start new chunk with potential overlap
                if overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                current_tokens += paragraph_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({"text": current_chunk.strip()})
        
        return chunks
    
    async def _semantic_chunking(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Semantic chunking using embeddings to group related content."""
        try:
            # First, get sentence chunks as base units
            sentence_chunks = await self._sentence_chunking(text, chunk_size // 2, 0)
            
            if len(sentence_chunks) <= 1:
                return sentence_chunks
            
            # Get embeddings for each sentence chunk
            model = await self._load_embedding_model()
            texts = [chunk["text"] for chunk in sentence_chunks]
            embeddings = model.encode(texts)
            
            # Group semantically similar chunks
            chunks = []
            current_group = [0]  # Start with first sentence
            current_tokens = self._count_tokens(texts[0])
            
            for i in range(1, len(texts)):
                sentence_tokens = self._count_tokens(texts[i])
                
                # Calculate semantic similarity with current group
                current_embedding = embeddings[current_group].mean(axis=0)
                similarity = self._cosine_similarity(current_embedding, embeddings[i])
                
                # Decide whether to add to current group or start new one
                should_group = (
                    similarity > 0.7 and  # High semantic similarity
                    current_tokens + sentence_tokens <= chunk_size
                )
                
                if should_group:
                    current_group.append(i)
                    current_tokens += sentence_tokens
                else:
                    # Finalize current group
                    group_text = " ".join(texts[j] for j in current_group)
                    chunks.append({"text": group_text})
                    
                    # Start new group
                    current_group = [i]
                    current_tokens = sentence_tokens
            
            # Add final group
            if current_group:
                group_text = " ".join(texts[j] for j in current_group)
                chunks.append({"text": group_text})
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}, falling back to sentence chunking")
            return await self._sentence_chunking(text, chunk_size, overlap)
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get overlap text from the end of the current chunk."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        overlap_token_ids = tokens[-overlap_tokens:]
        return self.tokenizer.decode(overlap_token_ids)
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))