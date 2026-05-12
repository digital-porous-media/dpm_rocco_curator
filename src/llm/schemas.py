from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class RubricItem:
    """One criterion from the evaluation rubric, with its score and explanation."""
    criterion: str
    score: float
    explanation: str = None
    
@dataclass
class EvaluatorOutput:
    """Structured output from DescriptionEvaluator"""
    total_score: float
    rubric_breakdown: List[RubricItem]
    comments: Optional[str] = None

class Citation(BaseModel):
    """Citation schema for each statemnt in the improved description."""
    statement: str = Field(description="The statement from the enhanced description.")
    source: str = Field(description="Source of the added information (original_description, uploaded_document, or user_feedback)") # Original description or uploaded document
    quote: str = Field(description="The exact quote or statement from the source.")
    doc_title: Optional[str] = Field(default=None, description="Document title (filename without extension) for uploaded_document sources")
    page: Optional[int] = Field(default=None, description="Page number in source document for uploaded_document sources")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index for uploaded_document sources")

class EditorOutput(BaseModel):
    """Output from the description editor"""
    original_text: str = Field(description="Original description text")
    suggested_text: str = Field(description="Improved description text")
    rationale: str = Field(description="Explanation of changes made")
    citation: List[Citation] = Field(default_factory=list, description="Citations for added information")
    context_used: List[Dict[str, Any]] = Field(default_factory=list, description="Metadata for retrieved context chunks (doc_title, page, chunk_index, snippet)")

class EditingSession(BaseModel):
    """Schema for saving/loading editing sessions"""
    metadata: Dict[str, Any] = Field(description="Session metadata")
    created_at: str = Field(description="ISO format timestamp of session creation")
    original_description: Optional[str] = Field(default=None, description="The original description being edited")
    current_description: Optional[str] = Field(default=None, description="The current version of the description")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="History of user feedback and assistant responses"
    )
    rubric: Dict[str, Any] = Field(description="Evaluation rubric used in this session")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration settings like use_rag and top_k_context"
    )
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the session"""
        summary = f"Session created: {self.created_at}\n"
        summary += f"Conversation turns: {len(self.conversation_history)}\n"
        
        if self.original_description:
            summary += f"Original description length: {len(self.original_description)} chars\n"
        
        if self.current_description:
            summary += f"Current description length: {len(self.current_description)} chars\n"
        
        summary += f"RAG enabled: {self.config.get('use_rag', False)}\n"
        
        return summary
    
@dataclass
class PDFChunk:
    """A single text chunk extracted from a PDF, optionally with its embedding vector."""
    chunk_id: str
    text: str
    embedding: Optional[List[float]] = None
    source_pdf: Optional[str] = None