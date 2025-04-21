```markdown
# Categorizer

## Categorizer is a simple python package which you can use to categorize your string records into predefined -nested- categories using the power of LLMs. 

---

## Features

- **Hierarchical Categories**  
  Define multi‑level category trees in `categories.yaml`. Parent/child relationships are enforced bidirectionally.

- **LLM‑Driven Categorization**  

  Use the power of LLMs to semantically categorize any string into predefined categories in a level‑by‑level fashion.



- **Meta‑pattern  based Categorization (optional)**  
  When dealing with categorizayion It is common to encounter some record groups which follow specific pattern and therefore allow us to use regex‑based patterns to categorize them. This package allows you to setup such egex‑based “classification_patterns” in `bank_patterns.yaml` (or your own file) for instant, rule‑based tagging.

- **Keyword Trigger based Categorization (optional)**  
  Naïve but fast auto‑trigger on keywords defined per category (“auto_trigger_keyword”).



- **Flexible Record I/O**  
  Load records from a Pandas DataFrame, Python list, or single string. Outputs a DataFrame with selected categories, rationale, and more.

- **Prompt Templating & Pipelines**  
  Customize prompt order and post‑processing via pipeline stages (e.g., “SemanticIsolation”).

- **Extensible & Async‑Ready**  
  Easily extend `MyLLMService` for new operations. Supports async translation & classification endpoints.

---

## Installation

```bash
pip install categorizer
```

Or clone & install locally:

```bash
git clone https://github.com/karaposu/categorizer.git
cd categorizer
pip install .
```


---
# Usage 
## Initial Configuration



1. **Define your Categories **  
   Edit `categorizer/categories.yaml` to define your category tree. 
   keyword_identifier field is used for keyword_trigger. 
   ```yaml
   
   - Finance:
       helpers:
         keyword_identifier: ["invoice", "payment"]
         text_rules_for_llm: []
         description: "All finance‑related records"
       subcategories:
         - Revenue:
             helpers:
               keyword_identifier: ["sale", "subscription"]
               text_rules_for_llm: []
               description: ""
         - Expense:
             helpers:
               keyword_identifier: ["purchase", "refund"]
               text_rules_for_llm: []
               description: ""
   ```

2. **(Optional) Define your Meta‑Patterns**  
   Edit `categorizer/bank_patterns.yaml` under `meta_patterns.<owner>.classification_patterns` to add regex rules for already known pattern clusters in your dataset
   ```yaml
   meta_patterns:
     default:
       classification_patterns:
         - pattern: "(?i)refund"
           lvl1: Expense
           lvl2: Refund
   ```

3. **Upload your records and run the categorization**  
  ```python

  import pandas as pd
  from categorizer.record_manager import RecordManager

  # Sample DataFrame
  df = pd.DataFrame([
      {"text": "Dinner at Gray House café", "record_id": 1},
      {"text": "Electricity bill from VVC",   "record_id": 2},
  ])

  # Initialize manager
  rm = RecordManager(debug=True)

  # Load & categorize
  rm.load_records(df, categories_yaml_path="categorizer/categories.yaml")
  result_df = rm.categorize_records()

  print(result_df)
  ```


---

## Quick Start

### 1. CategorizationEngine (Standalone)

```python
from categorizer.categorization_engine import CategorizationEngine
from categorizer.record import Record

# Initialize engine
engine = CategorizationEngine(subcategory_level=2, debug=True)

# Create a Record
rec = Record.from_string(
    text="Subscription payment to Netflix",
    record_id=123,
    categories="categorizer/categories.yaml"
)

# Run regex & keyword first, then LLM fallback
engine.categorize_record(rec, use_metapattern=True, use_keyword=True)

print("Level 1:", rec.lvl1.name)
print("Level 2:", rec.lvl2.name)
print("By:", rec.categorized_by)
```

### 2. RecordManager (Batch)

```python
import pandas as pd
from categorizer.record_manager import RecordManager

# Sample DataFrame
df = pd.DataFrame([
    {"text": "Dinner at Gray House café", "record_id": 1},
    {"text": "Electricity bill from VVC",   "record_id": 2},
])

# Initialize manager
rm = RecordManager(debug=True)

# Load & categorize
rm.load_records(df, categories_yaml_path="categorizer/categories.yaml")
result_df = rm.categorize_records()

print(result_df)
```

---



## License

MIT © Enes Kuzucu  
```