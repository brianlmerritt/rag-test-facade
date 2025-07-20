from typing import Dict, Any
import asyncio
import json
import csv
import io
import structlog
from bs4 import BeautifulSoup
import PyPDF2
import pandas as pd

logger = structlog.get_logger()

class DocumentParser:
    """Document parser for various file formats."""
    
    async def parse_file(self, file_path: str, format: str) -> str:
        """Parse document from file path."""
        if format == "pdf":
            return await self._parse_pdf(file_path)
        elif format == "docx":
            return await self._parse_docx(file_path)
        else:
            # Read as text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return await self.parse_text(content, format)
    
    async def parse_text(self, content: str, format: str) -> str:
        """Parse document from text content."""
        if format == "txt" or format == "md":
            return content
        elif format == "html":
            return await self._parse_html(content)
        elif format == "json":
            return await self._parse_json(content)
        elif format == "csv":
            return await self._parse_csv(content)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def _parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text_content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
            
            return text_content.strip()
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    async def _parse_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document
            doc = Document(file_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            return "\n".join(text_content)
        except ImportError:
            raise ValueError("python-docx package required for DOCX parsing")
        except Exception as e:
            logger.error(f"DOCX parsing failed: {e}")
            raise ValueError(f"Failed to parse DOCX: {str(e)}")
    
    async def _parse_html(self, content: str) -> str:
        """Extract text from HTML content."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"HTML parsing failed: {e}")
            raise ValueError(f"Failed to parse HTML: {str(e)}")
    
    async def _parse_json(self, content: str) -> str:
        """Extract text from JSON content."""
        try:
            data = json.loads(content)
            return await self._extract_text_from_json(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            logger.error(f"JSON processing failed: {e}")
            raise ValueError(f"Failed to process JSON: {str(e)}")
    
    async def _extract_text_from_json(self, data: Any, max_depth: int = 10) -> str:
        """Recursively extract text from JSON data."""
        if max_depth <= 0:
            return str(data)
        
        if isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                # Include key as context
                text_parts.append(f"{key}: {await self._extract_text_from_json(value, max_depth - 1)}")
            return " | ".join(text_parts)
        elif isinstance(data, list):
            text_parts = []
            for item in data:
                text_parts.append(await self._extract_text_from_json(item, max_depth - 1))
            return " | ".join(text_parts)
        else:
            return str(data)
    
    async def _parse_csv(self, content: str) -> str:
        """Extract text from CSV content."""
        try:
            # Use pandas for robust CSV parsing
            df = pd.read_csv(io.StringIO(content))
            
            # Convert to text representation
            text_parts = []
            
            # Add column headers
            text_parts.append("Columns: " + " | ".join(df.columns))
            
            # Add rows
            for index, row in df.iterrows():
                row_text = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                text_parts.append(f"Row {index + 1}: {row_text}")
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"CSV parsing failed: {e}")
            raise ValueError(f"Failed to parse CSV: {str(e)}")