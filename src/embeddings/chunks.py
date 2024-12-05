from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

class DocumentChunker:
    """
    A class used to split documents into chunks using RecursiveCharacterTextSplitter.
    """
    def __init__(self, directory: str, chunk_size: int = 800, chunk_overlap: int = 50):
        """
        Initialize the DocumentChunker with specified chunk size and overlap.

        :param chunk_size: The maximum size of each chunk in characters.
        :param chunk_overlap: The number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        self.directory = directory
        
    def read_doc(self):
        """
        Read documents from the specifique directory.

        :return: A list documents.
        """
        file_loader=PyPDFDirectoryLoader(self.directory)
        documents=file_loader.load()
        return documents
    
    def chunk_data(self) -> List[Any]:
        """
        Splits the given documents into chunks.

        :return: A list of chunked documents.
        """
        split_docs = self.text_splitter.split_documents(self.read_doc())
        return split_docs

# Usage example:
# docs = [...]  # Assume this is a list of documents
# chunker = DocumentChunker()
# documents = chunker.chunk_data(docs)
