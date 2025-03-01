import json
import time
import os
import re
from tqdm import tqdm
import logging
from dotenv import load_dotenv
from openai import OpenAI
from src.model import DeepSeekModel
import numpy as np
from scipy import stats

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GSM8KBenchmark:
    def __init__(self, model=None):
        """
        Initialize the benchmark with a model
        
        Args:
            model: An instance of DeepSeekModel or compatible model
        """
        self.model = model if model else DeepSeekModel()
        
        # Initialize OpenAI client for evaluation
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized for evaluation")
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not found. Evaluation with GPT-4o will not be available.")
    
    def load_questions(self, file_path="./data/gsm8k_selected_questions.json"):
        """
        Load questions from a JSON file
        
        Args:
            file_path: Path to the JSON file containing questions
            
        Returns:
            List of question dictionaries
        """
        with open(file_path, "r") as f:
            return json.load(f)
    
    def create_prompt(self, question):
        """
        Create a prompt for the model based on the question
        
        Args:
            question: The question text
            
        Returns:
            A formatted prompt string
        """
        return f"""Question: {question}

Please solve this math problem step-by-step. Put your final answer within \\boxed{{}}.\n<think>\n"""
    
    def extract_reference_answer(self, answer_text):
        """
        Extract the numerical answer from GSM8K reference answer (after ####)
        
        Args:
            answer_text: The reference answer text from GSM8K
            
        Returns:
            The extracted numerical answer or None if not found
        """
        # Find the part after ####
        if "####" in answer_text:
            final_part = answer_text.split("####")[-1].strip()
            # Extract the number from this part
            numbers = re.findall(r'-?\d+\.?\d*', final_part)
            if numbers:
                return numbers[0]
        
        # Fallback: Get the last number in the text
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def extract_model_answer(self, model_output):
        """
        Extract the numerical answer from the model's output (within \boxed{})
        
        Args:
            model_output: The model's generated text
            
        Returns:
            The extracted numerical answer or None if not found
        """
        # Try to find the answer within \boxed{}
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', model_output)
        if boxed_match:
            boxed_content = boxed_match.group(1)
            # Extract number from the boxed content
            numbers = re.findall(r'-?\d+\.?\d*', boxed_content)
            if numbers:
                return numbers[0]
        
        # Fallback: Look for explicit "the answer is X" pattern
        answer_match = re.search(r'[Tt]he\s+answer\s+is\s+(-?\d+\.?\d*)', model_output)
        if answer_match:
            return answer_match.group(1)
        
        # Final fallback: Get the last number in the text
        numbers = re.findall(r'-?\d+\.?\d*', model_output)
        if numbers:
            return numbers[-1]
        
        return None
    
    def evaluate_problem(self, problem):
        """
        Evaluate a single problem using the model
        
        Args:
            problem: Problem dictionary with 'question' and 'answer' keys
            
        Returns:
            Dictionary with evaluation results
        """
        question = problem["question"]
        start_time = time.time()  # Start timing
        
        try:
            # Create prompt
            prompt = self.create_prompt(question)
            
            # Generate answer using the model
            model_output = self.model.generate(prompt)
            
            # Extract reference and model answers for direct comparison
            reference_answer = self.extract_reference_answer(problem["answer"])
            model_answer = self.extract_model_answer(model_output)
            
            # Check if answers match
            numerical_match = False
            if reference_answer and model_answer:
                try:
                    # Convert to float for numerical comparison (handles formatting differences)
                    numerical_match = float(reference_answer) == float(model_answer)
                except ValueError:
                    # If conversion fails, fall back to string comparison
                    numerical_match = reference_answer == model_answer
            
            result = {
                "question": question,
                "reference_answer": problem["answer"],
                "extracted_reference_answer": reference_answer,
                "model_output": model_output,
                "extracted_model_answer": model_answer,
                "numerical_match": numerical_match,
                "difficulty": problem.get("difficulty", 0)  # Include difficulty from problem data
            }
            
            # If OpenAI client is available, evaluate the reasoning
            if self.openai_client:
                try:
                    # Set a timeout for the OpenAI API call
                    eval_result = self.evaluate_reasoning_with_gpt4o(result, timeout=30)
                    result.update(eval_result)
                except Exception as e:
                    logger.error(f"Error in GPT-4o evaluation: {e}")
                    result["reasoning_consistent"] = False
                    result["reasoning_explanation"] = f"Error during evaluation: {str(e)}"
            
            # Add duration
            result["duration"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            # Handle errors and still record duration
            duration = time.time() - start_time
            logger.error(f"Error evaluating problem: {e}")
            return {
                "question": question,
                "reference_answer": problem["answer"],
                "error": str(e),
                "numerical_match": False,
                "difficulty": problem.get("difficulty", 0),
                "duration": duration
            }
    
    def evaluate_reasoning_with_gpt4o(self, result, timeout=30):
        """
        Evaluate the model's reasoning process (not the final answer) using GPT-4o
        
        Args:
            result: Dictionary containing question, model_output, etc.
            timeout: Timeout in seconds for the API call
            
        Returns:
            Dictionary with reasoning evaluation results from GPT-4o
        """
        evaluation_prompt = f"""
You are an expert mathematics evaluator. Your task is to evaluate ONLY the REASONING PROCESS in a solution to a math problem, not the final numerical answer.

QUESTION:
{result['question']}

MODEL'S SOLUTION:
{result['model_output']}

Focus only on whether the model's REASONING is consistent and mathematically sound. Ignore any numerical errors in calculation that don't affect the overall reasoning approach.

Please provide your verdict as "True" if the reasoning is logical and consistent (even if there are small calculation errors), or "False" if the reasoning has fundamental flaws or incorrect approaches.

After your True/False verdict, provide a brief explanation of your evaluation. Use the format:

VERDICT: [True/False]
EXPLANATION: [Your explanation of why the reasoning is consistent or flawed]
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert mathematics evaluator focusing only on reasoning quality."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                timeout=timeout  # Add explicit timeout
            )
            
            # Extract evaluation from GPT-4o response
            evaluation_text = response.choices[0].message.content.strip()
            
            # Parse the verdict and explanation
            verdict_match = re.search(r'VERDICT:\s*(True|False)', evaluation_text, re.IGNORECASE)
            explanation_match = re.search(r'EXPLANATION:\s*(.*)', evaluation_text, re.DOTALL)
            
            is_reasoning_consistent = False
            explanation = "Could not parse GPT-4o response"
            
            if verdict_match:
                is_reasoning_consistent = verdict_match.group(1).lower() == "true"
            
            if explanation_match:
                explanation = explanation_match.group(1).strip()
            
            return {
                "reasoning_consistent": is_reasoning_consistent,
                "reasoning_explanation": explanation,
                "gpt4o_full_response": evaluation_text
            }
            
        except Exception as e:
            logger.error(f"Error evaluating reasoning with GPT-4o: {e}")
            raise  # Re-raise to be caught by the calling function
    
    def run_benchmark(self, questions):
        """
        Run benchmark on a list of questions
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            List of evaluation results
        """
        results = []
        total_start_time = time.time()  # Track total benchmark time
        
        # Evaluate each question
        for question in tqdm(questions, desc="Evaluating problems"):
            result = self.evaluate_problem(question)
            results.append(result)
            # Log duration for monitoring
            logger.info(f"Problem evaluated in {result.get('duration', 0):.2f} seconds.")
        
        # Calculate accuracy metrics
        numerical_correct = sum(1 for r in results if r.get("numerical_match", False))
        numerical_accuracy = numerical_correct / len(results) if results else 0
        
        logger.info(f"Numerical Match Accuracy: {numerical_accuracy:.2%}")
        
        if self.openai_client and results and "reasoning_consistent" in results[0]:
            reasoning_correct = sum(1 for r in results if r.get("reasoning_consistent", False))
            reasoning_accuracy = reasoning_correct / len(results)
            logger.info(f"Reasoning Consistency Rate: {reasoning_accuracy:.2%}")
        
        # Log total time
        total_time = time.time() - total_start_time
        logger.info(f"Total benchmark time: {total_time:.2f} seconds")
        logger.info(f"Average time per problem: {total_time/len(results):.2f} seconds")
        
        return results
    
    def save_results(self, results, output_file="./src/results/gsm8k_benchmark_results.txt"):
        """
        Save benchmark results to a file
        
        Args:
            results: List of evaluation results
            output_file: Path to output file
        """
        # Calculate overall metrics
        numerical_correct = sum(1 for r in results if r.get("numerical_match", False))
        numerical_accuracy = numerical_correct / len(results) if results else 0
        
        reasoning_accuracy = None
        if results and "reasoning_consistent" in results[0]:
            reasoning_correct = sum(1 for r in results if r.get("reasoning_consistent", False))
            reasoning_accuracy = reasoning_correct / len(results)
        
        # Calculate average duration
        durations = [r.get("duration", 0) for r in results]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        with open(output_file, "w") as f:
            f.write(f"GSM8K Benchmark Results\n")
            f.write(f"Model: DeepSeek-R1-Distill-Qwen-1.5B\n\n")
            f.write(f"Number of problems evaluated: {len(results)}\n")
            f.write(f"Numerical Match Accuracy: {numerical_accuracy:.2%}\n")
            f.write(f"Average Duration: {avg_duration:.2f} seconds\n")
            
            if reasoning_accuracy is not None:
                f.write(f"Reasoning Consistency Rate: {reasoning_accuracy:.2%}\n")
            
            f.write("\n")
            
            for i, result in enumerate(results):
                f.write(f"Example {i+1}:\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Model Output:\n{result['model_output']}\n")
                f.write(f"Reference Answer: {result.get('extracted_reference_answer', 'N/A')}\n")
                f.write(f"Model Answer: {result.get('extracted_model_answer', 'N/A')}\n")
                f.write(f"Numerical Match: {result.get('numerical_match', False)}\n")
                f.write(f"Duration: {result.get('duration', 0):.2f} seconds\n")
                f.write(f"Difficulty: {result.get('difficulty', 0)}\n")
                
                if "reasoning_consistent" in result:
                    f.write(f"Reasoning Consistent: {result['reasoning_consistent']}\n")
                    f.write(f"Reasoning Explanation: {result['reasoning_explanation']}\n")
                
                f.write("-" * 50 + "\n\n")
        
        # Also save results in JSON format for easier analysis
        json_output_file = output_file.replace(".txt", ".json")
        with open(json_output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Detailed results saved to {output_file} and {json_output_file}")

    def run_robustness_benchmark(self, questions, num_runs=10, output_dir="./src/results/robustness", injection=None):
        """
        Run the benchmark multiple times on each question to test robustness,
        then use majority voting to determine the final answer.
        
        Args:
            questions: List of question dictionaries
            num_runs: Number of times to run each question
            output_dir: Directory to save incremental results
            injection: Optional text to inject as the start of reasoning
            
        Returns:
            List of robustness evaluation results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directories for full outputs
        full_outputs_dir = os.path.join(output_dir, "full_outputs")
        os.makedirs(full_outputs_dir, exist_ok=True)
        
        # Initialize results list
        robustness_results = []
        
        # Save paths
        json_results_path = os.path.join(output_dir, "gsm8k_robustness_results.json")
        txt_results_path = os.path.join(output_dir, "gsm8k_robustness_results.txt")
        
        # Load existing results if available
        if os.path.exists(json_results_path):
            try:
                with open(json_results_path, 'r') as f:
                    robustness_results = json.load(f)
                logger.info(f"Loaded {len(robustness_results)} existing robustness results")
                
                # Determine which questions we've already processed
                processed_questions = set()
                for result in robustness_results:
                    processed_questions.add(result.get('question', ''))
                
                # Filter out questions we've already processed
                original_count = len(questions)
                questions = [q for q in questions if q['question'] not in processed_questions]
                logger.info(f"Skipping {original_count - len(questions)} already processed questions")
            except Exception as e:
                logger.warning(f"Error loading existing results, starting fresh: {e}")
                robustness_results = []
        
        # Process each question
        for question_index, question in enumerate(questions):
            logger.info(f"Testing robustness for question {question_index+1}/{len(questions)}")
            
            # Store all generated answers for this question
            question_answers = []
            question_obj = question["question"]
            reference_answer = self.extract_reference_answer(question["answer"])
            
            # Create a safe filename for this question
            question_id = f"question_{len(robustness_results)}"
            
            # Run the model multiple times on this question
            for run in tqdm(range(num_runs), desc=f"Question {question_index+1}", leave=False):
                # Create the base prompt
                base_prompt = self.create_prompt(question_obj)
                
                # If injection is provided, append it to the prompt
                if injection:
                    # Find where to inject (after "<think>\n" if it exists)
                    if "<think>" in base_prompt:
                        prompt_parts = base_prompt.split("<think>\n", 1)
                        prompt = prompt_parts[0] + "<think>\n" + injection + "\n" + prompt_parts[1] if len(prompt_parts) > 1 else base_prompt + injection
                    else:
                        prompt = base_prompt + "\n" + injection
                else:
                    prompt = base_prompt
                
                model_output = self.model.generate(prompt)
                model_answer = self.extract_model_answer(model_output)
                
                # Store the answer
                question_answers.append({
                    "run": run + 1,
                    "model_output": model_output,
                    "extracted_answer": model_answer,
                    "prompt": prompt  # Store the full prompt including injection
                })
            
            # Perform majority voting
            answer_counts = {}
            for run_result in question_answers:
                if run_result["extracted_answer"]:
                    answer = run_result["extracted_answer"]
                    answer_counts[answer] = answer_counts.get(answer, 0) + 1
            
            # Find the majority answer
            majority_answer = None
            max_count = 0
            
            for answer, count in answer_counts.items():
                if count > max_count:
                    majority_answer = answer
                    max_count = count
            
            # Calculate the consistency ratio
            consistency_ratio = max_count / num_runs if question_answers else 0
            
            # Check if the majority answer matches the reference
            is_correct = False
            if majority_answer and reference_answer:
                try:
                    # Compare as floats to handle formatting differences
                    is_correct = float(majority_answer) == float(reference_answer)
                except ValueError:
                    # Fall back to string comparison
                    is_correct = majority_answer == reference_answer
            
            # Create the robustness result
            robustness_result = {
                "question": question_obj,
                "reference_answer": reference_answer,
                "majority_answer": majority_answer,
                "is_correct": is_correct,
                "consistency_ratio": consistency_ratio,
                "answer_distribution": answer_counts,
                "num_runs": num_runs,
                "individual_runs": question_answers,
                "difficulty": question.get("difficulty", 0),
                "question_id": question_id,
                "injection_used": bool(injection),
                "injection_text": injection if injection else None
            }
            
            # Add to results list
            robustness_results.append(robustness_result)
            
            # Log the result
            logger.info(f"Question {question_index+1} result:")
            logger.info(f"  Majority answer: {majority_answer}")
            logger.info(f"  Reference answer: {reference_answer}")
            logger.info(f"  Correct: {is_correct}")
            logger.info(f"  Consistency ratio: {consistency_ratio:.2f}")
            
            # Save incremental results after each question
            try:
                # 1. Save full model outputs to a separate file
                full_output_path = os.path.join(full_outputs_dir, f"{question_id}.json")
                with open(full_output_path, "w") as f:
                    # Include question info and all full model outputs
                    full_data = {
                        "question": question_obj,
                        "reference_answer": reference_answer,
                        "difficulty": question.get("difficulty", 0),
                        "injection_used": bool(injection),
                        "injection_text": injection if injection else None,
                        "runs": [
                            {
                                "run": run["run"],
                                "extracted_answer": run["extracted_answer"],
                                "full_output": run["model_output"],
                                "prompt": run["prompt"]
                            }
                            for run in question_answers
                        ]
                    }
                    json.dump(full_data, f, indent=2)
                
                # 2. Save simplified JSON (without full model outputs) for the main results file
                simplified_results = []
                for result in robustness_results:
                    simplified_result = result.copy()
                    # Remove the full model outputs from individual runs
                    simplified_result["individual_runs"] = [
                        {
                            "run": run["run"],
                            "extracted_answer": run["extracted_answer"]
                        }
                        for run in result["individual_runs"]
                    ]
                    # Add reference to full outputs file
                    simplified_result["full_outputs_file"] = f"full_outputs/{simplified_result['question_id']}.json"
                    simplified_results.append(simplified_result)
                
                with open(json_results_path, "w") as f:
                    json.dump(simplified_results, f, indent=2)
                
                # 3. Update text results
                self._update_robustness_text_results(robustness_results, txt_results_path, injection)
                
                logger.info(f"Saved incremental results after question {question_index+1}")
                logger.info(f"Full model outputs saved to {full_output_path}")
            except Exception as e:
                logger.error(f"Error saving incremental results: {e}")
        
        # Calculate overall statistics
        if robustness_results:
            correct_count = sum(1 for r in robustness_results if r["is_correct"])
            accuracy = correct_count / len(robustness_results)
            avg_consistency = sum(r["consistency_ratio"] for r in robustness_results) / len(robustness_results)
            
            logger.info(f"Robustness benchmark complete:")
            logger.info(f"  Overall accuracy: {accuracy:.2%}")
            logger.info(f"  Average consistency ratio: {avg_consistency:.2%}")
            logger.info(f"  Full model outputs saved to {full_outputs_dir}")
        
        return robustness_results

    def _update_robustness_text_results(self, results, output_file, injection=None):
        """
        Update the text results file for robustness benchmark
        
        Args:
            results: List of robustness evaluation results
            output_file: Path to output file
            injection: Optional text that was injected into the prompt
        """
        # Calculate overall metrics
        correct_count = sum(1 for r in results if r["is_correct"])
        accuracy = correct_count / len(results) if results else 0
        avg_consistency = sum(r["consistency_ratio"] for r in results) / len(results) if results else 0
        
        with open(output_file, "w") as f:
            f.write(f"GSM8K Robustness Benchmark Results (Incremental)\n")
            f.write(f"Model: DeepSeek-R1-Distill-Qwen-1.5B\n\n")
            
            if injection:
                f.write(f"Using reasoning injection:\n")
                f.write(f"{injection[:200]}... (truncated)\n\n")
            
            f.write(f"Number of problems evaluated so far: {len(results)}\n")
            f.write(f"Number of runs per problem: {results[0]['num_runs'] if results else 0}\n")
            f.write(f"Overall accuracy: {accuracy:.2%}\n")
            f.write(f"Average consistency ratio: {avg_consistency:.2%}\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Example {i+1}:\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Reference answer: {result['reference_answer']}\n")
                f.write(f"Majority answer: {result['majority_answer']}\n")
                f.write(f"Is correct: {result['is_correct']}\n")
                f.write(f"Consistency ratio: {result['consistency_ratio']:.2f}\n")
                f.write(f"Difficulty: {result.get('difficulty', 0)}\n")
                
                # Write answer distribution
                f.write("Answer distribution:\n")
                for answer, count in result["answer_distribution"].items():
                    percentage = (count / result["num_runs"]) * 100
                    f.write(f"  {answer}: {count}/{result['num_runs']} ({percentage:.1f}%)\n")
                
                f.write("-" * 50 + "\n\n")

    def save_robustness_results(self, results, output_file="./src/results/robustness/gsm8k_robustness_results.txt"):
        """
        Save robustness benchmark results to a file
        
        Args:
            results: List of robustness evaluation results
            output_file: Path to output file
        """
        # Calculate overall metrics
        correct_count = sum(1 for r in results if r["is_correct"])
        accuracy = correct_count / len(results) if results else 0
        avg_consistency = sum(r["consistency_ratio"] for r in results) / len(results) if results else 0
        
        with open(output_file, "w") as f:
            f.write(f"GSM8K Robustness Benchmark Results\n")
            f.write(f"Model: DeepSeek-R1-Distill-Qwen-1.5B\n\n")
            f.write(f"Number of problems evaluated: {len(results)}\n")
            f.write(f"Number of runs per problem: {results[0]['num_runs'] if results else 0}\n")
            f.write(f"Overall accuracy: {accuracy:.2%}\n")
            f.write(f"Average consistency ratio: {avg_consistency:.2%}\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Example {i+1}:\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Reference answer: {result['reference_answer']}\n")
                f.write(f"Majority answer: {result['majority_answer']}\n")
                f.write(f"Is correct: {result['is_correct']}\n")
                f.write(f"Consistency ratio: {result['consistency_ratio']:.2f}\n")
                f.write(f"Difficulty: {result.get('difficulty', 0)}\n")
                
                # Write answer distribution
                f.write("Answer distribution:\n")
                for answer, count in result["answer_distribution"].items():
                    percentage = (count / result["num_runs"]) * 100
                    f.write(f"  {answer}: {count}/{result['num_runs']} ({percentage:.1f}%)\n")
                
                f.write("-" * 50 + "\n\n")
        
        # Also save results in JSON format for easier analysis
        json_output_file = output_file.replace(".txt", ".json")
        
        # Create a simplified version of the results for JSON
        # (without the full model outputs to keep file size manageable)
        simplified_results = []
        for result in results:
            simplified_result = result.copy()
            # Remove the full model outputs from individual runs
            simplified_result["individual_runs"] = [
                {
                    "run": run["run"],
                    "extracted_answer": run["extracted_answer"]
                }
                for run in result["individual_runs"]
            ]
            simplified_results.append(simplified_result)
        
        with open(json_output_file, "w") as f:
            json.dump(simplified_results, f, indent=2)
        
        logger.info(f"Robustness results saved to {output_file} and {json_output_file}")

    def analyze_duration_correctness(self, results):
        """
        Analyze the correlation between runtime duration and correctness.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Extract durations and correctness
        durations = []
        correctness = []
        difficulties = []
        
        for result in results:
            if 'duration' in result and 'numerical_match' in result:
                durations.append(result['duration'])
                correctness.append(1 if result['numerical_match'] else 0)
                difficulties.append(result.get('difficulty', 0))
        
        if not durations or len(durations) < 2:
            logger.warning("Not enough data points for correlation analysis")
            return {"error": "Not enough data points"}
        
        # Convert to numpy arrays for analysis
        durations = np.array(durations)
        correctness = np.array(correctness)
        difficulties = np.array(difficulties)
        
        # Calculate correlation between duration and correctness
        corr_coef, p_value = stats.pointbiserialr(correctness, durations)
        
        # Logistic regression of correctness on duration
        try:
            from sklearn.linear_model import LogisticRegression
            X = durations.reshape(-1, 1)
            model = LogisticRegression(solver='liblinear').fit(X, correctness)
            coef = model.coef_[0][0]
            intercept = model.intercept_[0]
        except ImportError:
            logger.warning("scikit-learn not available, skipping logistic regression")
            coef = None
            intercept = None
        
        # Calculate average durations for correct and incorrect answers
        correct_durations = durations[correctness == 1]
        incorrect_durations = durations[correctness == 0]
        
        avg_correct_duration = np.mean(correct_durations) if len(correct_durations) > 0 else 0
        avg_incorrect_duration = np.mean(incorrect_durations) if len(incorrect_durations) > 0 else 0
        
        # Calculate partial correlation controlling for difficulty
        if len(difficulties) > 0:
            # Remove the linear dependence of X and Y on Z
            try:
                slope_x, intercept_x = np.polyfit(difficulties, durations, 1)
                slope_y, intercept_y = np.polyfit(difficulties, correctness, 1)
                
                residuals_x = durations - (slope_x * difficulties + intercept_x)
                residuals_y = correctness - (slope_y * difficulties + intercept_y)
                
                partial_corr, p_partial = stats.pearsonr(residuals_x, residuals_y)
            except:
                logger.warning("Error calculating partial correlation")
                partial_corr = None
                p_partial = None
        else:
            partial_corr = None
            p_partial = None
        
        # Group durations into bins and calculate accuracy for each bin
        try:
            if len(durations) >= 5:  # Need at least a few data points
                num_bins = min(5, len(durations) // 2)  # Create reasonable number of bins
                duration_bins = np.linspace(min(durations), max(durations), num_bins+1)
                binned_accuracy = []
                bin_centers = []
                
                for i in range(num_bins):
                    bin_start = duration_bins[i]
                    bin_end = duration_bins[i+1]
                    bin_centers.append((bin_start + bin_end) / 2)
                    
                    # Find all points in this bin
                    in_bin = (durations >= bin_start) & (durations <= bin_end)
                    bin_correct = correctness[in_bin]
                    
                    if len(bin_correct) > 0:
                        bin_accuracy = np.mean(bin_correct)
                    else:
                        bin_accuracy = 0
                    
                    binned_accuracy.append(bin_accuracy)
        except:
            logger.warning("Error calculating binned accuracy")
            binned_accuracy = []
            bin_centers = []
        
        # Create analysis result
        analysis = {
            "correlation_coefficient": corr_coef,
            "p_value": p_value,
            "logistic_regression_coefficient": coef,
            "logistic_regression_intercept": intercept,
            "avg_correct_duration": avg_correct_duration,
            "avg_incorrect_duration": avg_incorrect_duration,
            "duration_difference_percentage": ((avg_correct_duration - avg_incorrect_duration) / avg_incorrect_duration * 100) if avg_incorrect_duration > 0 else 0,
            "partial_correlation_controlling_for_difficulty": partial_corr,
            "partial_correlation_p_value": p_partial,
            "binned_durations": bin_centers,
            "binned_accuracy": binned_accuracy,
            "correct_count": sum(correctness),
            "incorrect_count": len(correctness) - sum(correctness),
            "total_count": len(correctness)
        }
        
        # Log summary of analysis
        logger.info(f"Duration-Correctness Analysis:")
        logger.info(f"  Correlation coefficient: {corr_coef:.3f} (p-value: {p_value:.3f})")
        if partial_corr is not None:
            logger.info(f"  Partial correlation (controlling for difficulty): {partial_corr:.3f} (p-value: {p_partial:.3f})")
        logger.info(f"  Average duration for correct answers: {avg_correct_duration:.2f} seconds")
        logger.info(f"  Average duration for incorrect answers: {avg_incorrect_duration:.2f} seconds")
        if avg_incorrect_duration > 0:
            diff_pct = (avg_correct_duration - avg_incorrect_duration) / avg_incorrect_duration * 100
            logger.info(f"  Correct answers take {diff_pct:.1f}% {'longer' if diff_pct > 0 else 'shorter'} than incorrect ones")
        
        return analysis

    def save_duration_correctness_analysis(self, analysis, output_file="./src/results/baseline/duration_correctness_analysis.txt"):
        """
        Save duration-correctness analysis results to a file
        
        Args:
            analysis: Dictionary with analysis results
            output_file: Path to output file
        """
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w") as f:
            f.write("Duration vs Correctness Analysis\n")
            f.write("===============================\n\n")
            
            f.write(f"Total problems: {analysis.get('total_count', 0)}\n")
            f.write(f"Correct answers: {analysis.get('correct_count', 0)}\n")
            f.write(f"Incorrect answers: {analysis.get('incorrect_count', 0)}\n\n")
            
            f.write("Correlation Analysis:\n")
            f.write(f"  Correlation coefficient: {analysis.get('correlation_coefficient', 'N/A'):.3f}\n")
            f.write(f"  p-value: {analysis.get('p_value', 'N/A'):.3f}\n")
            
            if analysis.get('partial_correlation_controlling_for_difficulty') is not None:
                f.write("\nPartial Correlation (controlling for difficulty):\n")
                f.write(f"  Partial correlation: {analysis.get('partial_correlation_controlling_for_difficulty', 'N/A'):.3f}\n")
                f.write(f"  p-value: {analysis.get('partial_correlation_p_value', 'N/A'):.3f}\n")
            
            f.write("\nLogistic Regression:\n")
            coef = analysis.get('logistic_regression_coefficient')
            intercept = analysis.get('logistic_regression_intercept')
            if coef is not None and intercept is not None:
                f.write(f"  Coefficient: {coef:.4f}\n")
                f.write(f"  Intercept: {intercept:.4f}\n")
                f.write(f"  Interpretation: ")
                if coef > 0:
                    f.write(f"Longer duration increases probability of correctness\n")
                else:
                    f.write(f"Shorter duration increases probability of correctness\n")
            else:
                f.write("  Not calculated\n")
            
            f.write("\nDuration Statistics:\n")
            f.write(f"  Average duration for correct answers: {analysis.get('avg_correct_duration', 0):.2f} seconds\n")
            f.write(f"  Average duration for incorrect answers: {analysis.get('avg_incorrect_duration', 0):.2f} seconds\n")
            
            diff_pct = analysis.get('duration_difference_percentage', 0)
            f.write(f"  Correct answers take {abs(diff_pct):.1f}% {'longer' if diff_pct > 0 else 'shorter'} than incorrect ones\n")
        
        # Convert NumPy types to Python native types for JSON serialization
        json_safe_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, np.ndarray):
                json_safe_analysis[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                json_safe_analysis[key] = int(value)
            elif isinstance(value, (np.float64, np.float32, np.float16)):
                json_safe_analysis[key] = float(value)
            else:
                json_safe_analysis[key] = value
        
        # Save as JSON for easier plotting
        json_output_file = output_file.replace(".txt", ".json")
        import json
        with open(json_output_file, "w") as f:
            json.dump(json_safe_analysis, f, indent=2)
        
        logger.info(f"Duration-correctness analysis saved to {output_file} and {json_output_file}")

# Example usage (this would typically be in a separate main script)
if __name__ == "__main__":
    # Initialize the benchmark
    benchmark = GSM8KBenchmark()
    
    # Load questions
    questions = benchmark.load_questions()
    logger.info(f"Loaded {len(questions)} questions")
    
    # Run benchmark
    logger.info("Running benchmark on GSM8K questions...")
    results = benchmark.run_benchmark(questions)
    
    # Save results
    benchmark.save_results(results)
    
    # Plot durations if plotting utility is available
    try:
        from src.utils.plotting import plot_durations_by_difficulty
        plot_durations_by_difficulty(
            results_file="./src/results/gsm8k_benchmark_results.json",
            output_file="./src/results/duration_vs_difficulty.png"
        )
    except ImportError:
        logger.warning("Plotting utility not available. Install matplotlib to enable plotting.")
    
    # Display sample results
    logger.info("\nExample Problems and Solutions:")
    for i, result in enumerate(results[:3]):  # Show first 3 examples
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Question: {result['question']}")
        logger.info(f"Model Answer: {result.get('extracted_model_answer', 'N/A')}")
        logger.info(f"Reference Answer: {result.get('extracted_reference_answer', 'N/A')}")
        logger.info(f"Numerical Match: {result.get('numerical_match', False)}")
        logger.info(f"Duration: {result.get('duration', 0):.2f} seconds")
        logger.info(f"Difficulty: {result.get('difficulty', 0)}")
        
        if "reasoning_consistent" in result:
            logger.info(f"Reasoning Consistent: {result['reasoning_consistent']}")
            logger.info(f"Explanation: {result['reasoning_explanation'][:100]}...")
