# here is myllmservice.py


import logging

# def get_module_logger():
#     if __name__ == '__main__':
#         module_name = __spec__.name
#     else:
#         module_name = __name__
#     return logging.getLogger(module_name)
#
# logger = get_module_logger()



# logger = logging.getLogger(__name__)
import asyncio
from llmservice.base_service import BaseLLMService
from llmservice.generation_engine import GenerationRequest, GenerationResult
from typing import Optional, Union


# add default model param to init.

class MyLLMService(BaseLLMService):
    def __init__(self, logger=None, max_concurrent_requests=5):
        super().__init__(
            # logger=logger,
            logger=logging.getLogger(__name__),
            default_model_name="gpt-4o-mini",
            yaml_file_path='prompts.yaml',
            max_rpm=60,
            max_concurrent_requests=max_concurrent_requests,
        )
        # No need for a semaphore here, it's handled in BaseLLMService



    def categorize_with_parent(self,
                               record: str,
                               list_of_classes,
                               parent_category: str,
                               request_id: Optional[Union[str, int]] = None) -> GenerationResult:



        data_for_placeholders = {
            'list_of_classes': list_of_classes,
            'parent_category': parent_category,  # Using the merged string
            'string_record_to_be_classified': record,
        }


        order = ["list_of_classes", "parent_category", "string_record_to_be_classified", "category_task_description"]

        unformatted_prompt = self.generation_engine.craft_prompt(data_for_placeholders, order)

        generation_request = GenerationRequest(
            data_for_placeholders=data_for_placeholders,
            unformatted_prompt=unformatted_prompt,
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
        data_for_placeholders = {
            'list_of_classes': list_of_classes,

            'string_record_to_be_classified': record,
        }

        order = ["list_of_classes",  "string_record_to_be_classified", "category_task_description"]

        unformatted_prompt = self.generation_engine.craft_prompt(data_for_placeholders, order)

        pipeline_config = [
            {
                'type': 'SemanticIsolation',
                'params': {
                    'semantic_element_for_extraction': 'pure category'
                }
            }
            # {
            #     'type': 'ConvertToDict',
            #     'params': {}
            # },
            # {
            #     'type': 'ExtractValue',
            #     'params': {'key': 'answer'}  # Extract the 'answer' key from the dictionary
            # }
        ]

        generation_request = GenerationRequest(
            data_for_placeholders=data_for_placeholders,
            unformatted_prompt=unformatted_prompt,
            model="gpt-4o-mini",
            output_type="str",
            # use_string2dict=False,
            operation_name="categorize_simple",
            pipeline_config= pipeline_config,
            request_id=request_id
        )

        # self.logger.info("now will enter execute_generation...(myllmservice)")

        # for i in range(3):

        generation_result = self.execute_generation(generation_request)

        # self.logger.info("came back to llmservice...(myllmservice)")
            # if generation_result.

        return generation_result




    async def translate_to_russian_async(self, input_paragraph: str, request_id: Optional[Union[str, int]] = None) -> GenerationResult:
        # Concurrency control is handled in BaseLLMService
        data_for_placeholders = {'input_paragraph': input_paragraph}
        order = ["input_paragraph", "translate_to_russian"]

        unformatted_prompt = self.generation_engine.craft_prompt(data_for_placeholders, order)

        generation_request = GenerationRequest(
            data_for_placeholders=data_for_placeholders,
            unformatted_prompt=unformatted_prompt,
            model="gpt-4o-mini",
            output_type="str",

            operation_name="translate_to_russian",
            request_id=request_id
        )

        # Execute the generation asynchronously
        generation_result = await self.execute_generation_async(generation_request)
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
