# Categorizer


## Categorizer is a simple tool which you can use to categorise your string records into predefined -nested- categories using the power of LLMs. 

- upload your categories and subcategories ( read from yaml file or create them on the fly )
- initialize LEC. You can use different modes and your own task-specific prompts if you like
- Run the LEC and it will output a dataframe with all categories and subcategories and llm's reasoning to select them
- It is possible to leave notes for LLM for each category to help him categorize with more accuracy
- You can also use included naive classification method which supports regex based or keyword matching mechanism to reduce the LLM compute

Here is some benchmarking you to understand better. 

 category depth
 category combination size
 allowed retry

| Number of Records | Main Model      | Refiner Model   | Categorization Mode | Batch Prompting | Accuracy | Total Time | Avg Token | CPU Type           | GPU Type         |
|-------------------|-----------------|-----------------|----------------------|-----------------|----------|------------|-----------|--------------------|------------------|
| 1000              | Model A         | Refiner X       | Mode 1               | Yes             | 92.5%    | 10 mins    | 512       | Intel Xeon E5-2670 | NVIDIA Tesla K80 |
| 2000              | Model B         | Refiner Y       | Mode 2               | No              | 89.0%    | 20 mins    | 1024      | Intel Xeon E5-2680 | NVIDIA Tesla V100|
| 5000              | Model C         | Refiner Z       | Mode 3               | Yes             | 94.7%    | 50 mins    | 768       | AMD EPYC 7742      | NVIDIA A100      |
| 10000             | Model D         | Refiner W       | Mode 4               | No              | 88.3%    | 1 hr 40 mins| 2048     | Intel Xeon E5-2690 | NVIDIA RTX 3090  |

## Usage 

```pip install lec

lec = LLMEnhancedClassifier(
        llm_model=llm_model,
        llm_refiner_model=llm_refiner,
        categories_yaml_path='categories.yaml',
        meta_patterns_yaml_path='bank_patterns.yaml',
        subcategory_level=2  # Change this value to set the number of subcategories (max 4)
    )
    
lec.load_records(df)
 df = lec.classify_lvl_by_lvl()
```