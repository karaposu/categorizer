# here is myllmservice.py


import logging

# logger = logging.getLogger(__name__)
import asyncio
from llmservice.base_service import BaseLLMService
from llmservice.generation_engine import GenerationRequest, GenerationResult
from typing import Optional, Union


# add default model param to init.

class MyLLMService(BaseLLMService):
    def __init__(self, logger=None, max_concurrent_requests=200):
        super().__init__(
            logger=logging.getLogger(__name__),
            default_model_name="gpt-4o-mini",
            max_rpm=500,
            max_concurrent_requests=max_concurrent_requests,
        )
        # No need for a semaphore here, it's handled in BaseLLMService



    def categorize_with_parent(self,
                               record: str,
                               list_of_classes,
                               parent_category: str,
                               request_id: Optional[Union[str, int]] = None) -> GenerationResult:


        formatted_prompt = f"""Here is list of classes: {list_of_classes},
                            and here is parent_category : {parent_category}
                            and here is string record to be classified {record}

                            Task Description:
                            Identify the Category: Determine which of the categories the string belongs to.
                            Extra Information - Helpers:  There might be additional information under each subcategory labeled as 'helpers'. These helpers include descs for the taxonomy,  but
                            should be considered as extra information and not directly involved in the classification task
                            Instructions:
                            Given the string record, first identify the category of the given string using given category list,  (your final answer shouldnt include words like "likely").
                            Use the 'Helpers' section for additional context.  And also at the end explain your reasoning in a very short way. 
                            Make sure category is selected from given categories and matches 100%
                            Examples:
                            Record: "Jumper Cable"
                            lvl1: interconnectors
                            
                            Record: "STM32"
                            lvl1: microcontrollers
                             """
    
        generation_request = GenerationRequest(
           
            formatted_prompt=formatted_prompt,
            model="gpt-4o",
            output_type="str",
            operation_name="categorize_with_parent",
            request_id=request_id
        )

        generation_result = self.execute_generation(generation_request)


        return generation_result

    def categorize_simple(self,
                               record: str,
                               list_of_classes,
                               request_id: Optional[Union[str, int]] = None) -> GenerationResult:
        
        formatted_prompt = f"""Here is list of classes: {list_of_classes},

                            and here is string record to be classified {record}

                            Task Description:
                            Identify the Category: Determine which of the categories the string belongs to.
                            Extra Information - Helpers:  There might be additional information under each subcategory labeled as 'helpers'. These helpers include descs for the taxonomy,  but
                            should be considered as extra information and not directly involved in the classification task
                            Instructions:
                            Given the string record, first identify the category of the given string using given category list,  (your final answer shouldnt include words like "likely").
                            Use the 'Helpers' section for additional context.  And also at the end explain your reasoning in a very short way. 
                            Make sure category is selected from given categories and matches 100%
                            Examples:
                            Record: "Jumper Cable"
                            lvl1: interconnectors
                            
                            Record: "STM32"
                            lvl1: microcontrollers
                             """


        pipeline_config = [
            {
                'type': 'SemanticIsolation',
                'params': {
                    'semantic_element_for_extraction': 'pure category'
                }
            }
        ]

        generation_request = GenerationRequest(
            formatted_prompt=formatted_prompt,
            model="gpt-4o-mini",
            output_type="str",
            operation_name="categorize_simple",
            pipeline_config= pipeline_config,
            request_id=request_id
        )

        generation_result = self.execute_generation(generation_request)

    
        return generation_result



def main():
    """
    Main function to test the categorize_simple method of MyLLMService.
    """
    # Initialize the service
    my_llm_service = MyLLMService()

    # Sample data for testing
    sample_record = "The company reported a significant increase in revenue this quarter."
    sample_classes = ["Finance", "Marketing", "Operations", "Human Resources"]
    request_id = 1

    try:
        # Perform categorization
        result = my_llm_service.categorize_simple(
            record=sample_record,
            list_of_classes=sample_classes,
            request_id=request_id
        )

        # Print the result
        print("Generation Result:", result)
        if result.success:
            print("Categorized Content:", result.content)
        else:
            print("Error:", result.error_message)
    except Exception as e:
        print(f"An exception occurred: {e}")


if __name__ == "__main__":
    main()
