from abc import ABC, abstractmethod
from typing import List, Union
from langchain_core.documents import Document

class BaseIngestor(ABC):
    """Base Document ingestor class"""
    @abstractmethod
    def ingest(self, filepath: Union[str, List[str]]) -> List[Document]:
        """Ingest document from URL and return list of Document objects"""
        pass




