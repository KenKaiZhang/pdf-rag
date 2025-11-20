import pdfplumber
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def extract_text_and_tables(self, pdf_path: str) -> Tuple[List[Dict], List[Dict]]:
        text_blocks = []
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    text_blocks.append({
                        "content": text,
                        "page": page_num,
                        "type": "text"
                    })
            
                page_tables = page.extract_tables()
                for table_num, table in enumerate(page_tables, start=1):
                    if table:
                        table_text = self._format_table(table)
                        tables.append({
                            "content": table_text,
                            "page": page_num,
                            "table_number": table_num,
                            "type": "table",
                            "raw_data": table
                        })
        
        return text_blocks, tables
    
    def _format_table(self, table: List[List]) -> str:
        if not table:
            return ""
        
        headers = table[0]
        rows = table[1:]
        
        formatted = f"Table with columns: {", ".join(str(h) for h in headers if h)}\n\n"
        
        for row in rows:
            row_text = " | ".join(str(cell) if cell else "" for cell in row)
            formatted += row_text + "\n"
        
        return formatted
    
    def chunk_documents(self, text_blocks: List[Dict], tables: List[Dict]) -> List[Document]:
        documents = []
        
        for block in text_blocks:
            chunks = self.text_splitter.split_text(block["content"])
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "page": block["page"],
                        "type": "text",
                        "chunk_index": i
                    }
                )
                documents.append(doc)
                
        for table in tables:
            doc = Document(
                page_content=table["content"],
                metadata={
                    "page": table["page"],
                    "type": "table",
                    "table_number": table["table_number"]
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        text_blocks, tables = self.extract_text_and_tables(pdf_path)
        documents = self.chunk_documents(text_blocks, tables)
        return documents
            