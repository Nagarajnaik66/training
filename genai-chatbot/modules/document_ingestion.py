import aiofiles
import asyncio
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from modules.utils import config, logger

SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.pptx', '.xlsx', '.docx']

class DocumentLoader:
    @staticmethod
    async def load_and_split_async(file_path: str) -> List[Document]:
        try:
            # Asynchronously read the file
            async with aiofiles.open(file_path, mode='rb') as file:
                content = await file.read()
                
            # Initialize the loader
            loader = UnstructuredFileLoader(file_path)
            
            # Define a text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config['app']['chunk_size'],
                chunk_overlap=config['app']['chunk_overlap']
            )
            
            # Load and split the document
            chunks = loader.load_and_split(text_splitter)
            return chunks
        except Exception as e:
            logger.error(f"Error loading and splitting {file_path}: {str(e)}")
            raise