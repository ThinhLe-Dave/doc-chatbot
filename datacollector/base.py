from abc import ABC, abstractmethod
from typing import List, Optional


class DataCollector(ABC):
    """Abstract base class for data collectors."""
    
    def __init__(self, output_file: str = "data.json"):
        self.output_file = output_file
        self.documents: List = []
    
    @abstractmethod
    def collect(self, source: str, **kwargs) -> List:
        """Collect data from source and return list of documents."""
        pass
    
    @abstractmethod
    def export_to_json(self, output_file: str = None) -> str:
        """Export collected documents to JSON file."""
        pass
    
    def get_document_count(self) -> int:
        """Return number of collected documents."""
        return len(self.documents)
    
    def build_chunks(self, output_file: str = None) -> tuple:
        """Build chunk cache from collected documents. Override if direct chunk support is available."""
        raise NotImplementedError("Subclasses with chunk support must implement this method.")