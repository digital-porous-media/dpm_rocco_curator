from dataclasses import dataclass, field
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
    rationale: str
    citation: Optional[List[Citation]] = field(default_factory=list)
@dataclass
class Citation:
    statement: str
    source: str  # Original description or context chunk
    quote: str  # Support statement
    
@dataclass
class PDFChunk:
    chunk_id: str
    text: str
    embedding: Optional[List[float]] = None
    source_pdf: Optional[str] = None