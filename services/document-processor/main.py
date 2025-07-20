from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import asyncio
import base64
import json
import os
import tempfile
import uuid
from datetime import datetime
import structlog

from processors.chunking import ChunkingService
from processors.document_parser import DocumentParser
from utils.metrics import setup_metrics
from utils.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

app = FastAPI(
    title="Document Processor Service",
    description="OpenAI-compatible document processing and chunking service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup metrics
metrics_registry = setup_metrics(app)

# Initialize services
chunking_service = ChunkingService()
document_parser = DocumentParser()

class ChunkingOptions(BaseModel):
    chunk_size: int = Field(default=512, ge=1, le=4096)
    overlap: int = Field(default=50, ge=0, le=500)
    strategy: str = Field(default="semantic", regex="^(semantic|fixed|sentence|paragraph)$")

class DocumentProcessRequest(BaseModel):
    input: str = Field(..., description="Base64 encoded content or raw text")
    format: str = Field(..., regex="^(pdf|txt|html|json|csv|docx|md)$")
    model: str = Field(default="chunker-v1")
    encoding_format: str = Field(default="float", regex="^(float|base64)$")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    options: Optional[ChunkingOptions] = Field(default_factory=ChunkingOptions)

class DocumentChunk(BaseModel):
    object: str = Field(default="document.chunk")
    index: int
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class DocumentProcessResponse(BaseModel):
    object: str = Field(default="list")
    data: List[DocumentChunk]
    model: str
    usage: Dict[str, int]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/v1/documents/process", response_model=DocumentProcessResponse)
async def process_document(request: DocumentProcessRequest):
    """
    Process a document and return chunks with metadata.
    Compatible with OpenAI API structure.
    """
    try:
        start_time = datetime.utcnow()
        
        # Parse document content
        if request.format in ["pdf", "docx"]:
            # Decode base64 content for binary formats
            try:
                content_bytes = base64.b64decode(request.input)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {str(e)}")
            
            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=f".{request.format}", delete=False) as temp_file:
                temp_file.write(content_bytes)
                temp_file_path = temp_file.name
            
            try:
                text_content = await document_parser.parse_file(temp_file_path, request.format)
            finally:
                os.unlink(temp_file_path)
        else:
            # Text-based formats
            if request.format in ["json", "csv"]:
                try:
                    decoded_content = base64.b64decode(request.input).decode('utf-8')
                except:
                    decoded_content = request.input
            else:
                decoded_content = request.input
            
            text_content = await document_parser.parse_text(decoded_content, request.format)
        
        # Create chunks
        chunks = await chunking_service.create_chunks(
            text=text_content,
            strategy=request.options.strategy,
            chunk_size=request.options.chunk_size,
            overlap=request.options.overlap,
            metadata=request.metadata
        )
        
        # Format response
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **request.metadata,
                "chunk_index": i,
                "chunk_strategy": request.options.strategy,
                "source_format": request.format,
                "processing_timestamp": start_time.isoformat()
            }
            
            chunk_objects.append(DocumentChunk(
                index=i,
                text=chunk["text"],
                metadata=chunk_metadata,
                embedding=chunk.get("embedding")
            ))
        
        # Calculate usage metrics
        total_tokens = sum(len(chunk.text.split()) for chunk in chunk_objects)
        
        response = DocumentProcessResponse(
            data=chunk_objects,
            model=request.model,
            usage={
                "total_tokens": total_tokens,
                "chunks_created": len(chunk_objects)
            }
        )
        
        logger.info(
            "Document processed successfully",
            format=request.format,
            chunks_created=len(chunk_objects),
            total_tokens=total_tokens,
            processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
        )
        
        return response
        
    except Exception as e:
        logger.error("Document processing failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/v1/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    options: str = None
):
    """
    Upload and process a document file directly.
    """
    try:
        # Parse options if provided
        chunk_options = ChunkingOptions()
        if options:
            options_dict = json.loads(options)
            chunk_options = ChunkingOptions(**options_dict)
        
        # Read file content
        content = await file.read()
        
        # Determine format from filename
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ["pdf", "txt", "html", "json", "csv", "docx", "md"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
        
        # Create request
        if file_extension in ["pdf", "docx"]:
            input_content = base64.b64encode(content).decode('utf-8')
        else:
            input_content = content.decode('utf-8')
        
        request = DocumentProcessRequest(
            input=input_content,
            format=file_extension,
            metadata={"filename": file.filename, "content_type": file.content_type},
            options=chunk_options
        )
        
        return await process_document(request)
        
    except Exception as e:
        logger.error("File upload processing failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"File upload processing failed: {str(e)}")

@app.get("/v1/models")
async def list_models():
    """List available chunking models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "chunker-v1",
                "object": "model",
                "created": 1699000000,
                "owned_by": "rag-test-facade",
                "strategies": ["semantic", "fixed", "sentence", "paragraph"]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)