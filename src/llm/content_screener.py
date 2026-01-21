from src.llm.client import RoccoClient
import json

class ContentScreener:
    """Screen contents for usefulness"""
    def __init__(self, model: RoccoClient):
        self.model = model
    
    def screen_user_content(self, content: str, context: str = None) -> dict:
        """
        Screen user provided content
        
        Returns: {
            'is_valid': bool,
            'issues': list of issues found
            'confidence': float (0-1)
            'recommendation': str
        }
        """
        
        screening_prompt = f"""You are a content quality screener for scientific dataset descriptions.
        
        Evaluate the following user provided content for these issues:
        1. Is it relevant to improving a dataset description?
        2. Does it contain accurate scientific information?
        3. Is it respectful and constructive?
        4. Does it contain gibberish, nonsense, or irrelevant language?
        5. Is it derogatory or inappropriate?
        
        User Feedback:
        "{content}"
        
        {f"Context (current description): {context}" if context else ""}
        Content Screener - Determines if user feedback is appropriate for enhancement

        Flagging Strategy (flag_for_review):
        - Potential quality issues that need human judgment
        - Edge cases that might be valid but need verification
        - Feedback that could improve the description but has minor concerns

        Examples of things to FLAG:
        1. Vague or overly general feedback
        - "Make it better"
        - "Add more details" (without specifying what)
        
        2. Feedback that contradicts the original description
        - Original: "sandstone sample"
        - Feedback: "This is limestone"
        
        3. Feedback requesting major structural changes
        - "Rewrite the entire description"
        - "Change the focus completely"
        
        4. Feedback with unverified claims
        - "This dataset was used in Nature paper" (without citation)
        - "This is the most important dataset in the field"
        
        5. Feedback that adds subjective opinions
        - "This is the best experimental setup"
        - "Everyone should use this method"
        
        6. Feedback that's partially unclear
        - Mix of clear and vague statements
        - Some claims without context

        Examples of things to REJECT:
        1. Offensive, discriminatory, or inappropriate content
        2. Completely irrelevant feedback (e.g., "What's the weather?")
        3. Feedback that's just spam or gibberish
        4. Feedback attempting to inject malicious content

        Examples of things to ACCEPT:
        1. Clear, specific improvements
        - "Add the sample diameter: 10 mm"
        - "Specify the imaging resolution: 2.8 microns"
        
        2. Factual corrections
        - "The temperature was 25°C, not 20°C"
        
        3. Clarifications of existing information
        - "Expand on what 'micro-CT' means for general readers"
        
        4. Relevant supplemental information
        - "This dataset was used in Smith et al. 2023"
        
        5. Minor structural changes for specific improvements
        - "Rephrase the methodology section to make it more concise"
        - "Update the methodology section to include more details"
        - "Reword the conclusion for clarity"
        - "Expand on the general context of creation and research problem"

        Respond in the following JSON format:
        {{
            "is_relevant": bool,
            "is_accurate": bool,
            "is_respectful": bool,
            "is_coherent": bool,
            "issues": [list of specific issues found],
            "confidence": float between 0 and 1,
            "recommendation": "accept" | "reject" | "flag_for_review"
        }}
        
        Do not provide any additional text outside the JSON.
        """
        
        response = self.model.send_prompt(prompt=screening_prompt)
        # print(response)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            return {
                "is_relevant": False,
                "is_accurate": False,
                "is_respectful": False,
                "is_coherent": False,
                "issues": ["Unable to parse feedback"],
                "confidence": 0.5,
                "recommendation": "flag_for_review"
            }
        
        