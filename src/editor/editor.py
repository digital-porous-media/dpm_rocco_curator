from src.llm.client import RoccoClient
from src.llm.schemas import EditorOutput, EvaluatorOutput, Citation, EditingSession
from src.common.utils import _build_rubric, _build_evaluation_text
from src.retriever.retriever import VectorStoreManager
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

class DescriptionEditor:
    """Improves dataset descriptions"""

    def __init__(self, model: RoccoClient, rubric: Dict, vector_store_manager: Optional[VectorStoreManager] = None, use_rag: bool = True, top_k_context: int = 5):
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
            }
        )
        
        # Ensure directory exists
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using Pydantic's model_dump_json
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(session.model_dump_json(indent=2))
        
        print(f"Session saved to {filepath}")
        print(session.get_summary())
    
    def load_session(self, filepath: Path) -> None:
        """Load a session from a file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Session file not found: {filepath}")
        
        # Load and validate using Pydantic schema
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Validate with schema
        session = EditingSession(**session_data)
        
        # Restore session state
        self.conversation_history = session.conversation_history
        self.session_created_at = session.created_at
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
            }
        )
        return session.get_summary()
    
    def retrieve_context(self, query: str = None) -> List[str]:
        """Retrieve relevant context from related papers"""
        
        if not self.use_rag or self.vector_store_manager is None:
            return []
        
        try:
            results = self.vector_store_manager.similarity_search(query, k=self.top_k_context)
            context_snippets = [doc.page_content for doc in results]
            return context_snippets
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return []
        
    def generate_search_query(self, draft_evaluation: EvaluatorOutput, query_all: bool = True) -> List[str]:
        """Generate search queries based on evaluation feedback"""
        queries = []
        
        criterion_queries = {
            1: "data description",
            2: "research goals objectives study purpose motivation",
            3: "sample material porous media type lithology",
            4: "research problem research question hypothesis",
            5: "applications reuse reproducibility validation machine learning simulation",
            6: "methodology experimental setup x-ray imaging technique data collection image acquisition scanning image processing",
            7: "dataset structure organization file data format contents",
            8: "quality control validation verification calibration inspection",
            9: None,
            10: "keywords relevant concepts domain-specific terminology nomenclature"
        }
        
        if query_all:
            for criterion_id, query in criterion_queries.items():
                if query:
                    queries.append(query)
        
        else:
            for criterion_id, rubric_item in enumerate(draft_evaluation.rubric_breakdown, 1):
                score = rubric_item.score
                # TODO Handle different score weights
                if score <= 0.5:
                    query = criterion_queries.get(criterion_id)
                    if query:
                        queries.append(query)
        
        if not queries:
            print("No queries generated - skipping context retrieval.")
                        
        return queries
        
                
    def build_prompt(self, draft_text: str, draft_evaluation: EvaluatorOutput, context: Optional[List[str]] = None, user_feedback: Optional[str] = None) -> str:
        """Prepare prompt for improving the draft"""
        rubric_str = _build_rubric(self.rubric)
        eval_feedback = _build_evaluation_text(draft_evaluation)
        
        # Build conversation history
        history = ""
        if self.conversation_history:
            history = "\n## CONVERSATION HISTORY:\n"
            for message in self.conversation_history:
                if message["role"] == "user":
                    history += f"\n**User Feedback:**\n{message['content']}\n"
                elif message["role"] == "assistant":
                    history += f"\n**Previous Version:**\n{message['content']}\n"
                    if "rationale" in message:
                        history += f"**Rationale:** {message['rationale']}\n"
        
        # Provide user feedback
        if user_feedback:
            task_description = (
                "You are an expert data curator for the Digital Porous Media Portal continuing an interactive dataset description editing session.\n"
                "The user has provided feedback on the previous version of your dataset description.\n"
                "Your task: Refine the description based on their feedback while preserving other improvements.\n"
            )
            user_feedback_section = f"\n## NEW USER FEEDBACK:\n{user_feedback}\n"
        else:
            task_description = (
                "You are an expert data curator for the Digital Porous Media Portal starting a new dataset description editing session.\n"
                "Your task: Rewrite the description so it maximizes compliance with the rubric criteria, addressing reviewer concerns and using only information from the papers, if available. Retain strengths of the original description.\n"
            )
            user_feedback_section = ""
        
        system_instructions = (
        f"{task_description}"
        "Below is the full rubric for evaluating dataset descriptions.\n"
        "You are also given the original description, reviewer feedback on areas that need improvement, and relevant context from related papers.\n"

        "CRITICAL RULES:\n"
        "1. Only include information that is explicitly stated in the original description or the provided paper context\n"
        "2. Do not add generic statements like 'organized into files and folders' or 'documentation provided' unless specifically mentioned\n"
        "3. DO not invent methodological details, sample properties, or organizational information\n"
        "4. DO NOT use speculative or uncertain language such as:\n"
        "   - 'potentially', 'possibly', 'likely', 'may include', 'probably'\n"
        "   - 'though the exact... is not specified'\n"
        "   - 'among other things', 'and similar', 'etc.'\n"
        "   - 'various', 'multiple', 'several' (unless specific numbers are given)\n"
        "5. If information is missing, simply omit it - do not acknowledge gaps or speculate\n"
        "6. When improving clarity or structure, preserve all factual content from the original\n"
        "7. Maintain the original language and tone for your responses\n"
        "\n"
        )     

        context_str = ""
        if context:
            context_str = "\n\n---\n\n".join(context)
            rag_context = f"""## RELEVANT CONTEXT FROM RESEARCH PAPERS

            The following excerpts from related research papers may provide useful information.
            Each chunk is numbered for citation purposes (CONTEXT_CHUNK_1, CONTEXT_CHUNK_2, etc.):

            {context_str}

            IMPORTANT: Only use information from these excerpts if it is directly relevant and factual. Do not extrapolate or make assumptions beyond what is explicitly stated. 
            When you use information from these excerpts, you must cite the specific context or quote from the retrieved text in your response.
            """
        else:
            rag_context = "## NO ADDITIONAL CONTEXT AVAILABLE\n\n You must work only with the information provided in the original description. Do not add speculative or generic information.\n"
        
        prompt = f"""
        ## TASK:
            Improve the dataset description below based on the rubric and reviewer feedback.
            {system_instructions}
            
        ## EVALUATION RUBRIC:
            {rubric_str}
            
        ## ORIGINAL DESCRIPTION:
            {draft_text}
            
        ## REVIEWER FEEDBACK:
            {eval_feedback}
            
        {rag_context}
        {history}
        {user_feedback_section}
        
        ## CITATION REQUIREMENTS:
        For every statement in your improved description that:
        1. Adds new information not in the original description, OR
        2. Provides more specific details than the original

        You MUST provide a citation showing:
        - The exact statement from your improved description
        - The source (either "original_description", "context_chunk", or "user_feedback")
        - The exact quote from the source that supports this statement

        ## EXAMPLES OF PROPER CITATIONS:

        Example 1 - Adding methodology details:
        Statement: "Micro-CT imaging was performed at 2.5 μm resolution"
        Source: "context_chunk"
        Quote: "The sample was scanned using micro-CT at a voxel size of 2.5 micrometers"

        Example 2 - Adding sample information:
        Statement: "The Berea sandstone sample was obtained from a depth of 1200 meters"
        Source: "context_chunk"
        Quote: "Core samples were extracted from the Berea formation at 1200m depth"

        Example 3 - Clarifying from original:
        Statement: "The dataset contains raw and processed CT images"
        Source: "original_description"
        Quote: "includes both raw scans and segmented images"

        ## OUTPUT FORMAT
        Provide your response as a JSON object with this structure:
        {{
            "updated_description": [
                {{
                    "updated_description": "Your improved description here",
                    "rationale": "Brief explanation of key changes made",
                    "citations": [
                        {{
                            "statement": "Specific statement from your improved description",
                            "source": "original_description, context_chunk, or user_feedback",
                            "quote": "Exact quote from the source"
                        }}
                    ]
                }}
            ]
        }}
        
        Do not provide any additional text outside the JSON.
        """
        return prompt


    def enhance(self, draft_text: str, draft_evaluation: EvaluatorOutput, retrieve_context: bool = True, context_override: Optional[List[str]] = None, query_all_criterion: bool = True, user_feedback: Optional[str] = None) -> EditorOutput:
        """Improve the description draft using evaluation feedback and optional context from papers"""
        
        if self.original_description is None:
            self.original_description = draft_text
        
        if user_feedback:
            self.conversation_history.append({
                "role": "user",
                "content": user_feedback
            })
        
        
        if context_override:
            context = context_override
        elif retrieve_context and self.use_rag:
            queries = self.generate_search_query(draft_evaluation, query_all=query_all_criterion)
            context = []
            if queries:
                seen_content = set()
                
                for i, query in enumerate(queries):
                    query_context = self.retrieve_context(query)
                    
                    for content in query_context:
                        if content not in seen_content:
                            context.append(content)
                            seen_content.add(content)
        else:
            context = []
        
        prompt = self.build_prompt(draft_text, draft_evaluation, context, user_feedback=user_feedback)
        raw_resp = self.model.send_prompt(prompt)
        
        try:
            data = json.loads(raw_resp.strip())
            
            citations = []
            if "citations" in data["updated_description"][0]:
                for cit in data["updated_description"][0]["citations"]:
                    citations.append(Citation(
                        statement=cit["statement"],
                        source=cit["source"],
                        quote=cit["quote"]
                    ))
                    
            updated_desc = EditorOutput(
                original_text=draft_text,
                suggested_text=data["updated_description"][0]["updated_description"],
                rationale=data["updated_description"][0]["rationale"],
                citation=citations
            )
            
            self.conversation_history.append({
                "role": "assistant",
                "content": updated_desc.suggested_text,
                "rationale": updated_desc.rationale
            })
            
            self.current_description = updated_desc.suggested_text
            
        except Exception as e:
            print(e)
            updated_desc = EditorOutput(
                original_text = draft_text,
                suggested_text = f"Could not parse the response as JSON. Please check the prompt format.",
                rationale = f"Parsing error: {str(e)}",
                citation=[]
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
        self.conversation_history = []