class EvaluatorOutput:
    """Structured output from DescriptionEvaluator"""
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.total_score = None
        self.breakdown = []

class EditorOutput:
    """Structured output from DescriptionEnhancer"""
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.improved_draft = None
        self.edit_summary = []
        self.confidence = None