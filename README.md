# MiniProject2

This project repository is organized into several main folders and files. Below is an overview of the file structure and descriptions of what each part does.

## File Structure

```
MiniProject2/
├── .gitignore
├── README.md
├── notebooks/
│   └── Task2/
├── results/
│   └── task2_1/
├── task1/
│   ├── __pycache__/
│   ├── analyze_results.ipynb
│   ├── base_eval.ipynb
│   ├── data_helpers/
│   ├── downloaded_outputs/
│   ├── evaluate_finetuned.py
│   ├── evaluate_harmbench.ipynb
│   ├── finetuning_qwen3-4b.ipynb
│   ├── harmbench/
│   ├── modal/
│   ├── plotting.ipynb
│   └── updated_judge_scores.ipynb
```

## Directory & File Explanations

### `notebooks/`
- Contains Jupyter notebooks and resources for organizing experiments on Task 2.
  - `Task2_1/`: Contains two notebooks for Task 2.1. getresponses.ipynb includes pipeline for getting responses for prompts. Miniproject2.ipynb for analysis of n-grams.
  - `Task2_2/`: Contains Watermarking_AIS.ipynb for Task 2.2 pipeline. 

### `results/`
- Stores output results of Model responses.
  - `task2_1/`: A subfolder for results/output from a part of Task 2.1.

### `task1/`
- Main folder containing scripts, notebooks, and helper modules for Task 1.
  - `analyze_results.ipynb`: Analyze the misalignment rates
  - `base_eval.ipynb`: Evaluate misalignment rate in Qwen3-4B
  - `evaluate_finetuned.py`: Evaluate finetuned models on freeform questions
  - `evaluate_harmbench.ipynb`: Evaluating base and finetuned models on HarmBench
  - `finetuning_qwen3-4b.ipynb`: Fine-tuning process of the Qwen3-4b model.
  - `plotting.ipynb`: For generating visualizations/plots from results.
  - `updated_judge_scores.ipynb`: Exploring judgezoo for alignment scoring and testing different coherence prompts.
  - `data_helpers/`: Module(s) or scripts for data handling, loading, or preprocessing.
  - `downloaded_outputs/`: Directory for downloaded results of all finetuned and base models on freeform response
  - `harmbench/`: Results of harmbench evaluation
  - `modal/`: scripts to run finetune evaluation on modal

---
