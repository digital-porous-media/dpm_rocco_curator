from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RubricItem:
    criterion: str
    score: float
    explanation: str = None
    
@dataclass
class EvaluatorOutput:
    """Structured output from DescriptionEvaluator"""
    total_score: float
    rubric_breakdown: List[RubricItem]
    comments: Optional[str] = None


@dataclass
class EditorOutput:
    original_text: str
    suggested_text: str
    rationale: Optional[str] = None
    
@dataclass
class PDFChunk:
    chunk_id: str
    text: str
    embedding: Optional[List[float]] = None
    source_pdf: Optional[str] = None