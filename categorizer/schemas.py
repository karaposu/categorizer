# schemas.py

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Union, Literal , List
import pprint

def indent_text(text, indent):
    indentation = ' ' * indent
    return '\n'.join(indentation + line for line in text.splitlines())

@dataclass
class GenerationRequest:

    data_for_placeholders: Dict[str, Any]
    unformatted_prompt: str
    model: Optional[str] = None
    output_type: Literal["json", "str"] = "str"
    operation_name: Optional[str] = None
    request_id: Optional[Union[str, int]] = None
    number_of_retries: Optional[int] = None
    pipeline_config: List[Dict[str, Any]] = field(default_factory=list)
    fail_fallback_value: Optional[str] = None



@dataclass
class CategorizationResult:
    success: bool
    category_list: Optional[List[Any]] = field(default=None)
    rationale_dict: Optional[Dict[str, Any]] = field(default=None)
    matched_pattern: Optional[str] = field(default=None)
    raw_llm_answer: Optional[str] = field(default=None)
    matched_keyword: Optional[str] = field(default=None)
    categorized_by: Optional[str] = field(default=None)

dummy_result = CategorizationResult(
    success=True,
    category_list=[
        {"step": "Data Cleaning", "status": "Completed"},
        {"step": "Model Training", "status": "In Progress"},
    ],
    rationale_dict={
        "reason_1": "The data was noisy, requiring preprocessing.",
        "reason_2": {"error_code": 404, "description": "Missing values found in dataset."}
    },
    matched_pattern="temperature-related queries",
    raw_llm_answer="The temperature in Istanbul is 22°C with light showers.",
    matched_keyword="temperature",
    categorized_by="automated_pipeline"
)


    # def __str__(self):
    #     result = ["GenerationResult:"]
    #     for field_info in fields(self):
    #         field_name = field_info.name
    #         value = getattr(self, field_name)
    #         field_str = f"{field_name}:"
    #         if isinstance(value, (dict, list)):
    #             field_str += "\n" + indent_text(pprint.pformat(value, indent=4), 4)
    #         elif isinstance(value, str) and '\n' in value:
    #             # Multi-line string, indent each line
    #             field_str += "\n" + indent_text(value, 4)
    #         else:
    #             field_str += f" {value}"
    #         result.append(field_str)
    #     return "\n\n".join(result)


