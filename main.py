import logging
import os
import argparse
from src.benchmark.benchmark import GSM8KBenchmark

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run GSM8K benchmark or robustness tests")
    parser.add_argument("--mode", type=str, choices=["normal", "robustness"], default="normal",
                        help="Benchmark mode: normal (once per question) or robustness (multiple runs)")
    parser.add_argument("--injection", type=str, help="Initial reasoning to inject into the prompt", default=None)
    parser.add_argument("--injection-file", type=str, help="File containing initial reasoning to inject", default=None)
    parser.add_argument("--num-runs", type=int, help="Number of runs per question (for robustness mode)", default=10)
    parser.add_argument("--questions", type=str, help="Path to questions file", default="./data/gsm8k_selected_questions.json")
    parser.add_argument("--question-index", type=int, help="Only run for a specific question index", default=None)
    parser.add_argument("--output-dir", type=str, help="Output directory for results", default="./src/results/baseline")
    args = parser.parse_args()
    
    # Process injection text
    injection = None
    if args.injection:
        injection = args.injection
    elif args.injection_file:
        try:
            with open(args.injection_file, 'r') as f:
                injection = f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read injection file: {e}")
            return
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the benchmark
    benchmark = GSM8KBenchmark()
    
    # Load questions
    questions = benchmark.load_questions(args.questions)
    logger.info(f"Loaded {len(questions)} questions")
    
    # Filter to specific question if requested
    if args.question_index is not None:
        if 0 <= args.question_index < len(questions):
            questions = [questions[args.question_index]]
            logger.info(f"Running only for question index {args.question_index}: {questions[0]['question'][:50]}...")
        else:
            logger.error(f"Question index {args.question_index} is out of range (0-{len(questions)-1})")
            return
    
    # Run benchmark based on selected mode
    if args.mode == "normal":
        # Run normal benchmark (once per question)
        logger.info(f"Running normal benchmark for {len(questions)} questions...")
        if injection:
            logger.info(f"Using reasoning injection: {injection[:50]}...")
        
        results = benchmark.run_benchmark(questions)
        
        # Save results
        output_file = os.path.join(args.output_dir, "gsm8k_benchmark_results.txt")
        json_output_file = os.path.join(args.output_dir, "gsm8k_benchmark_results.json")
        benchmark.save_results(results, output_file)
        
        # Display summary
        correct_count = sum(1 for r in results if r.get("numerical_match", False))
        accuracy = correct_count / len(results) if results else 0
        
        logger.info(f"\nBenchmark Summary:")
        logger.info(f"Questions evaluated: {len(results)}")
        logger.info(f"Correct answers: {correct_count}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        
    else:  # robustness mode
        # Run robustness benchmark (multiple runs per question)
        logger.info(f"Running robustness benchmark ({args.num_runs} runs per question)...")
        if injection:
            logger.info(f"Using reasoning injection: {injection[:50]}...")
        
        results = benchmark.run_robustness_benchmark(
            questions, 
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            injection=injection
        )
        
        # Display summary
        correct_count = sum(1 for r in results if r["is_correct"])
        accuracy = correct_count / len(results) if results else 0
        avg_consistency = sum(r["consistency_ratio"] for r in results) / len(results) if results else 0
        
        logger.info(f"\nRobustness Benchmark Summary:")
        logger.info(f"Overall accuracy with majority voting: {accuracy:.2%}")
        logger.info(f"Average consistency ratio: {avg_consistency:.2%}")

if __name__ == "__main__":
    main()