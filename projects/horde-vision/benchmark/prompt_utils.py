from pydantic import BaseModel, Field, conlist
from typing import List, Optional
from enum import Enum
# This module provides a set of Pydantic models that validate the structure
# of the AI data enrichment assistant's JSON output.

# Model for the 'vqa' array item
class VQAItem(BaseModel):
    """
    Represents a single Visual Question-Answer pair.
    """
    question: str = Field(..., description="A visual question about the image.")
    answer: str = Field(..., description="The answer to the visual question.")

# Model for the 'ocr' object
class OCRItem(BaseModel):
    """
    Represents the OCR (Optical Character Recognition) result.
    If no text is present, instruction and answer should be None.
    """
    instruction: Optional[str] = Field(None, description="The instruction to read the text, or null if no text is present.")
    answer: Optional[str] = Field(None, description="The transcribed text, or null if no text is present.")

# Model for the 'reason' object
class ReasonItem(BaseModel):
    """
    Represents a complex reasoning question and its answer.
    """
    instruction: str = Field(..., description="A complex reasoning question requiring inference.")
    answer: str = Field(..., description="A logical answer based on visual evidence.")

# Model for the 'instruct_follow' object
class InstructFollowItem(BaseModel):
    """
    Represents a two-step instruction and its direct answer.
    """
    instruction: str = Field(..., description="A two-step instruction to locate and describe an element.")
    answer: str = Field(..., description="The direct answer completing the instruction.")

# Model for the 'info' object
class ImageQuality(str, Enum):
    """Defines the visual quality of the image."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW_BLURRY = "Low/Blurry"

class PrimarySubject(str, Enum):
    """Defines the main subject matter of the image."""
    PERSON_PEOPLE = "Person/People"
    ANIMALS = "Animal(s)"
    VEHICLES = "Vehicle(s)"
    FOOD = "Food"
    OBJECTS = "Object(s)"
    SCENERY_ARCHITECTURE = "Scenery/Architecture"
    OTHER = "Other"

class HumanPresence(str, Enum):
    """Describes the number of people present in the image."""
    NONE = "None"
    SINGLE_PERSON = "Single Person"
    SMALL_GROUP = "Small Group (2-5)"
    CROWD = "Crowd (>5)"

class FaceVisibility(str, Enum):
    """Describes the visibility of human faces."""
    CLEAR_FACES = "Clear Faces"
    OBSCURED_NO_FACES = "Obscured/No Faces"
    NOT_APPLICABLE = "Not Applicable"

class TextType(str, Enum):
    """Describes the type of text present in the image."""
    PRINTED = "Printed"
    HANDWRITTEN = "Handwritten"
    SIGNAGE_LOGO = "Signage/Logo"
    NOT_APPLICABLE = "Not Applicable"

    
class ContentProperties(BaseModel):
    """Groups metadata related to the primary content and subjects within the image."""
    quality: ImageQuality = Field(..., description="The visual quality of the image.")
    primary_subject: PrimarySubject = Field(..., description="The main subject matter of the image.")
    object_count: int = Field(..., ge=0, description="The estimated number of distinct, primary objects.")
    human_presence: HumanPresence = Field(..., description="Describes the number of people present.")
    face_visibility: FaceVisibility = Field(..., description="Describes the visibility of human faces.")

class TextProperties(BaseModel):
    """Groups all metadata related to text found in the image."""
    contains_text: bool = Field(..., description="True if any text is present in the image.")
    is_suitable_for_ocr: bool = Field(..., description="True if the text is clear enough for OCR tasks.")
    text_type: TextType = Field(..., description="The type of text present (e.g., printed, handwritten).")

class TaskSuitability(BaseModel):
    """Groups flags that indicate if the image is a good candidate for specific ML tasks."""
    is_suitable_for_counting: bool = Field(..., description="True if there are multiple, easily countable objects.")
    is_suitable_for_reasoning: bool = Field(..., description="True if the scene implies a story, action, or context requiring inference.")
    is_suitable_for_multi_step_instruction: bool = Field(..., description="True if the scene is complex enough for multi-step instructions.")


class ImageInfoItem(BaseModel):
    """
    A comprehensive Pydantic model to store structured metadata about an image.
    This model is designed to be easily filterable and useful for guiding ML fine-tuning processes.
    """
    content_properties: ContentProperties
    text_properties: TextProperties
    task_suitability: TaskSuitability

# Model for the 'caption' object
class CaptionItem(BaseModel):
    """
    Represents the detailed factual caption in Kazakh.
    """
    text: str = Field(..., description="A factual, detailed caption in Kazakh (5 sentences, 40-60 words).")

# Main Pydantic model for the entire JSON output
class DataEnrichmentResponse(BaseModel):
    """
    The main model representing the entire structured JSON output from the AI assistant.
    This model can be used to validate the output to ensure it matches the required schema.
    """
    caption: CaptionItem
    vqa: conlist(VQAItem, min_length=3, max_length=3)  # Enforces exactly 3 items
    ocr: OCRItem
    reason: ReasonItem
    instruct_follow: InstructFollowItem
    info: ImageInfoItem



batch_prompt = """
You are a meticulous AI Data Annotation Specialist in kazakh language. Your mission is to perform a comprehensive analysis of the provided image and generate a single, rich JSON data record. Your output must be structured, precise, and grounded in visual evidence.

**Analysis & Generation Protocol:**

**Step 1: Pre-Analysis & Metadata Population**
First, conduct a holistic analysis of the image to determine its core attributes. Use this analysis to populate the `info` object. Be methodical.
- **Content Properties:** Assess the image quality, identify the primary subject, estimate the object count, and characterize human presence and face visibility.
- **Text Properties:** Determine if text is present and assess its suitability for OCR.
- **Task Suitability:** Based on the scene's content and complexity, evaluate if the image is a good candidate for counting, reasoning, and multi-step instruction tasks.

**Step 2: Content Generation**
Based on your pre-analysis, generate the content for the remaining fields. The `info` block you just created should guide you. For example, if `is_suitable_for_counting` is true, one of your VQA pairs should be a counting question.

**Detailed Field Instructions:**

1.  **`caption`**:
    * **Language:** `Kazakh`
    * **Content:** Write a factual, detailed caption. Describe only what is directly visible. Avoid opinions, emotions, or inferences (e.g., no "beautiful" or "seems happy"). No 'Суретте' at the beginning.
    * **Structure:** Compose 4-6 distinct sentences, totaling 40-70 words, ensuring each sentence describes a different aspect (e.g., main subject, background, secondary objects, action).
	Keep in mind, that image is taken from folder called {topic_name}, include it into caption if it clearly matches the scene.
    
2.  **`vqa`**:
    * Generate an array of exactly three diverse visual question-answer pairs.
    * The questions must be answerable from the image alone.
    * The diversity should reflect the image's content (e.g., include counting if countable objects are present, color questions if colors are distinct, etc.).

3.  **`ocr`**:
    * If the `info.text_properties.is_suitable_for_ocr` flag is `true`, create an instruction to read the most prominent text and provide the exact transcription.
    * If `false`, set the `instruction` and `answer` fields to `null`.

4.  **`reason`**:
    * If the `info.task_suitability.is_suitable_for_reasoning` flag is `true`, generate one complex reasoning question that requires inference about context, causality, or temporal states (e.g., "What season is it and why?").
    * If `false`, create a simpler identification question and answer instead, and set `answer` to explain why deep reasoning isn't possible (e.g., "Question: What is the primary object? Answer: A blue cup. The image lacks sufficient context for a deeper reasoning task.").

5.  **`instruct_follow`**:
    * If `info.task_suitability.is_suitable_for_multi_step_instruction` is `true`, create a single, two-step instruction (e.g., "Identify the person on the left, then describe the color of their shirt.").
    * If `false`, create a simple one-step instruction.

6.  **`info`**:
    * Populate this object based on your Step 1 Pre-Analysis, adhering strictly to the allowed Enum values in the schema. This is the most critical metadata block.

**Respond ONLY with a single JSON object that validates against this Pydantic schema:**

```json
{{
  "caption": {{
    "text": "..."
  }},
  "vqa": [
    {{
      "question": "...",
      "answer": "..."
    }},
    ...
  ],
  "ocr": {{
    "instruction": "...",
    "answer": "..."
  }},
  "reason": {{
    "instruction": "...",
    "answer": "..."
  }},
  "instruct_follow": {{
    "instruction": "...",
    "answer": "..."
  }},
  "info": {{
    "content_properties": {{
      "quality": "High" | "Medium" | "Low/Blurry",
      "primary_subject": "Person/People" | "Animal(s)" | "Vehicle(s)" | "Food" | "Object(s)" | "Scenery/Architecture" | "Other",
      "object_count": <integer>,
      "human_presence": "None" | "Single Person" | "Small Group (2-5)" | "Crowd (>5)",
      "face_visibility": "Clear Faces" | "Obscured/No Faces" | "Not Applicable"
    }},
    "text_properties": {{
      "contains_text": <boolean>,
      "is_suitable_for_ocr": <boolean>,
      "text_type": "Printed" | "Handwritten" | "Signage/Logo" | "Not Applicable"
    }},
    "task_suitability": {{
      "is_suitable_for_counting": <boolean>,
      "is_suitable_for_reasoning": <boolean>,
      "is_suitable_for_multi_step_instruction": <boolean>
    }}
  }}
}}
"""