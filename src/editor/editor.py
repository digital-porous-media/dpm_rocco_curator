from src.llm.client import RoccoClient
from src.llm.schemas import EditorOutput, EvaluatorOutput, Citation, EditingSession
from src.common.utils import _build_rubric, _build_evaluation_text
from src.retriever.retriever import VectorStoreManager
from src.prompts.loader import load_prompt, render
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
import json
from pathlib import Path


class DescriptionEditor:
    """Improves dataset descriptions"""

    def __init__(
        self,
        model: RoccoClient,
        rubric: Dict,
        vector_store_manager: Optional[VectorStoreManager] = None,
        use_rag: bool = True,
        top_k_context: int = 5,
    ):
        self.model = model
        self.rubric = rubric
        self.vector_store_manager = vector_store_manager
        self.use_rag = use_rag
        self.top_k_context = top_k_context
        self.conversation_history: List[Dict[str, str]] = []
        self.session_creation_time = datetime.now().isoformat()
        self.original_description = None
        self.current_description = None

    # TODO: Add logging

    def save_session(self, filepath: Path) -> None:
        """Save the current session to a file"""
        session = EditingSession(
            created_at=self.session_creation_time,
            original_description=self.original_description,
            current_description=self.current_description,
            conversation_history=self.conversation_history,
            rubric=self.rubric,
            config={
                "use_rag": self.use_rag,
                "top_k_context": self.top_k_context,
            },
        )

        # Ensure directory exists
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save using Pydantic's model_dump_json
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(session.model_dump_json(indent=2))

        print(f"Session saved to {filepath}")
        print(session.get_summary())

    def load_session(self, filepath: Path) -> None:
        """Load a session from a file"""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Session file not found: {filepath}")

        # Load and validate using Pydantic schema
        with open(filepath, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        # Validate with schema
        session = EditingSession(**session_data)

        # Restore session state
        self.conversation_history = session.conversation_history
        self.session_creation_time = session.created_at
        self.original_description = session.original_description
        self.current_description = session.current_description
        self.conversation_history = session.conversation_history

        # Restore config
        if session.config:
            self.use_rag = session.config.get("use_rag", self.use_rag)
            self.top_k_context = session.config.get("top_k_context", self.top_k_context)

        print(f"Session loaded from {filepath}")
        print(session.get_summary())

    def get_session_summary(self) -> str:
        """Get a summary of the current session"""
        session = EditingSession(
            created_at=self.session_metadata["created_at"],
            original_description=self.session_metadata.get("original_description"),
            current_description=self.session_metadata.get("current_description"),
            conversation_history=self.conversation_history,
            rubric=self.rubric,
            config={
                "use_rag": self.use_rag,
                "top_k_context": self.top_k_context,
            },
        )
        return session.get_summary()

    def retrieve_context(self, query: str = None) -> List[Document]:
        """Retrieve relevant context from related papers"""

        if not self.use_rag or self.vector_store_manager is None:
            return []

        try:
            results = self.vector_store_manager.similarity_search(
                query, k=self.top_k_context
            )
            return results
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return []

    def generate_search_query(
        self, draft_evaluation: EvaluatorOutput, query_all: bool = True
    ) -> List[str]:
        """Generate search queries based on evaluation feedback"""
        queries = []

        criterion_queries = {
            1: "data description data summary data overview",
            2: "research goals objectives study purpose motivation",
            3: "sample material porous media type lithology",
            4: "research problem research question hypothesis",
            5: "applications reuse reproducibility validation machine learning simulation",
            6: "methodology experimental setup x-ray imaging technique data collection image acquisition scanning image processing",
            7: "dataset structure organization file data format contents",
            8: "quality control validation verification calibration inspection",
            9: None,
            10: "keywords relevant concepts domain-specific terminology nomenclature",
        }

        if query_all:
            for criterion_id, query in criterion_queries.items():
                if query:
                    queries.append(query)

        else:
            for criterion_id, rubric_item in enumerate(
                draft_evaluation.rubric_breakdown, 1
            ):
                score = rubric_item.score
                # TODO Handle different score weights
                if score <= 0.5:
                    query = criterion_queries.get(criterion_id)
                    if query:
                        queries.append(query)

        if not queries:
            print("No queries generated - skipping context retrieval.")

        return queries

    def build_prompt(
        self,
        draft_text: str,
        draft_evaluation: EvaluatorOutput,
        context: Optional[Union[List[Document], List[str]]] = None,
        user_feedback: Optional[str] = None,
        history_override: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Prepare prompt for improving the draft"""
        rubric_str = _build_rubric(self.rubric)
        eval_feedback = _build_evaluation_text(draft_evaluation)

        # Build conversation history
        history = ""
        history_source = (
            history_override
            if history_override is not None
            else self.conversation_history
        )
        if history_source:
            history = "\n## CONVERSATION HISTORY:\n"
            for message in history_source:
                if message["role"] == "user":
                    history += f"\n**User Feedback:**\n{message['content']}\n"
                elif message["role"] == "assistant":
                    history += f"\n**Previous Version:**\n{message['content']}\n"
                    if "rationale" in message:
                        history += f"**Rationale:** {message['rationale']}\n"

        # Determine mode (initial vs refinement)
        mode = "refinement" if user_feedback else "initial"

        # Format context with separators and citation metadata
        context_str = ""
        if context:
            formatted_chunks = []
            for chunk_num, item in enumerate(context, 1):
                if isinstance(item, Document):
                    # Format Document with citation-ready metadata
                    chunk_header = f"[CONTEXT_CHUNK_{chunk_num}]"
                    metadata_lines = [chunk_header]

                    # Add source metadata if available
                    if item.metadata:
                        doc_title = item.metadata.get("doc_title", "unknown")
                        page = item.metadata.get("page")
                        chunk_idx = item.metadata.get("chunk_index")

                        source_info = f"Source: {doc_title}"
                        if page is not None:
                            source_info += f", Page {page}"
                        if chunk_idx is not None:
                            source_info += f", Chunk {chunk_idx}"
                        metadata_lines.append(source_info)

                    metadata_lines.append("---")
                    formatted_chunks.append(
                        "\n".join(metadata_lines) + "\n" + item.page_content
                    )
                else:
                    # String fallback (for context_override)
                    formatted_chunks.append(f"[CONTEXT_CHUNK_{chunk_num}]\n---\n{item}")

            context_str = "\n\n---\n\n".join(formatted_chunks)

        # Load prompt template and render
        prompt_data = load_prompt("editor")
        prompt = render(
            prompt_data["user"],
            mode=mode,
            context_str=context_str,
            rubric_str=rubric_str,
            original_description=draft_text,
            evaluation_feedback=eval_feedback,
            history=history,
            user_feedback=user_feedback or "",
        )
        return prompt

    def enhance(
        self,
        draft_text: str,
        draft_evaluation: EvaluatorOutput,
        retrieve_context: bool = True,
        context_override: Optional[List[str]] = None,
        query_all_criterion: bool = True,
        user_feedback: Optional[str] = None,
        history_override: Optional[List[Dict[str, str]]] = None,
    ) -> EditorOutput:
        """Improve the description draft using evaluation feedback and optional context from papers"""

        if self.original_description is None:
            self.original_description = draft_text

        if user_feedback:
            self.conversation_history.append({"role": "user", "content": user_feedback})

        context_metadata = []

        if context_override:
            # context_override is still a list of formatted strings
            context = context_override
        elif retrieve_context and self.use_rag:
            queries = self.generate_search_query(
                draft_evaluation, query_all=query_all_criterion
            )
            context = []
            if queries:
                seen_content = set()

                for i, query in enumerate(queries):
                    query_context = self.retrieve_context(query)

                    for doc in query_context:
                        if doc.page_content not in seen_content:
                            context.append(doc)
                            seen_content.add(doc.page_content)
                            context_metadata.append(
                                {
                                    "doc_title": doc.metadata.get(
                                        "doc_title", "unknown"
                                    ),
                                    "page": doc.metadata.get("page"),
                                    "chunk_index": doc.metadata.get("chunk_index"),
                                    "snippet": doc.page_content[:120],
                                }
                            )
        else:
            context = []

        prompt = self.build_prompt(
            draft_text,
            draft_evaluation,
            context,
            user_feedback=user_feedback,
            history_override=history_override,
        )
        raw_resp = self.model.send_prompt(prompt)

        try:
            data = json.loads(raw_resp.strip())

            citations = []
            if "citations" in data["updated_description"][0]:
                for cit in data["updated_description"][0]["citations"]:
                    citations.append(
                        Citation(
                            statement=cit["statement"],
                            source=cit["source"],
                            quote=cit["quote"],
                            doc_title=cit.get("doc_title"),
                            page=cit.get("page"),
                            chunk_index=cit.get("chunk_index"),
                        )
                    )

            updated_desc = EditorOutput(
                original_text=draft_text,
                suggested_text=data["updated_description"][0]["updated_description"],
                rationale=data["updated_description"][0]["rationale"],
                citation=citations,
                context_used=context_metadata,
            )

            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": updated_desc.suggested_text,
                    "rationale": updated_desc.rationale,
                }
            )

            self.current_description = updated_desc.suggested_text

        except Exception as e:
            print(e)
            updated_desc = EditorOutput(
                original_text=draft_text,
                suggested_text=f"Could not parse the response as JSON. Please check the prompt format.",
                rationale=f"Parsing error: {str(e)}",
                citation=[],
            )
        return updated_desc

    def print_enhancement_result(self, editor_output: EditorOutput) -> None:
        """Utility to print enhancement results"""
        print(f"Original Description:\n{editor_output.original_text}\n")
        print(f"Enhanced Description:\n{editor_output.suggested_text}\n")
        print(f"Justifications:\n {editor_output.rationale}")
        print(f"Citations:\n")
        for item in editor_output.citation:
            print(f"Statement: {item.statement}")
            print(f"Source: {item.source}")
            print(f"Source Quote: {item.quote}\n")

    def reset_conversation_history(self):
        """Clear all stored conversation turns, starting a fresh refinement session."""
        self.conversation_history = []
