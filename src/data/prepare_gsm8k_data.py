import random
import json
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def select_and_save_questions(num_samples=10, output_file="./data/gsm8k_selected_questions.json"):
    """
    Load the GSM8K dataset, select random questions, and save them to a file.
    """
    logging.info("Loading GSM8K dataset...")
    gsm8k = load_dataset("openai/gsm8k", "main")
    test_set = gsm8k["test"]
    
    # Randomly sample from the dataset
    logger.info(f"Selecting {num_samples} random questions...")
    indices = random.sample(range(len(test_set)), num_samples)
    selected_samples = [test_set[i] for i in indices]
    
    # Convert to serializable format
    selected_data = [
        {
            "id": idx,
            "question": sample["question"],
            "answer": sample["answer"]
        }
        for idx, sample in enumerate(selected_samples)
    ]
    
    # Save to file
    with open(output_file, "w") as f:
        json.dump(selected_data, f, indent=2)
    
    logger.info(f"Selected questions saved to {output_file}")
    
    return output_file

if __name__ == "__main__":
    select_and_save_questions(num_samples=10) 