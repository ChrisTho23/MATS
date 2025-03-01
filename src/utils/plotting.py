import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from scipy.special import expit  # For logistic function

logger = logging.getLogger(__name__)

def plot_durations_by_difficulty(results_file="./src/results/gsm8k_benchmark_results.json", output_file="./src/results/duration_vs_difficulty.png"):
    """
    Plot the duration of each evaluation against the difficulty of the problem.
    
    Args:
        results_file: Path to the JSON file containing benchmark results
        output_file: Path to save the plot
    """
    try:
        # Load the results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extract durations and difficulties
        durations = []
        difficulties = []
        questions = []
        
        for result in results:
            if 'duration' in result and 'difficulty' in result:
                durations.append(result['duration'])
                difficulties.append(result['difficulty'])
                questions.append(result['question'][:30] + '...')  # Truncate for display
        
        if not durations:
            logger.error("No duration or difficulty data found in results file")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Sort data by difficulty for the first plot (scatter plot)
        sorted_indices = np.argsort(difficulties)
        sorted_difficulties = [difficulties[i] for i in sorted_indices]
        sorted_durations = [durations[i] for i in sorted_indices]
        
        # Create scatter plot of durations vs difficulties
        ax1.scatter(sorted_difficulties, sorted_durations, alpha=0.7, s=100)
        ax1.set_xlabel('Difficulty (10=easiest, 1=hardest)')
        ax1.set_ylabel('Duration (seconds)')
        ax1.set_title('Evaluation Duration vs Problem Difficulty')
        ax1.grid(True)
        
        # Add best fit line
        if len(sorted_difficulties) > 1:
            z = np.polyfit(sorted_difficulties, sorted_durations, 1)
            p = np.poly1d(z)
            ax1.plot(sorted_difficulties, p(sorted_difficulties), "r--", alpha=0.8)
            correlation = np.corrcoef(sorted_difficulties, sorted_durations)[0, 1]
            ax1.text(0.05, 0.95, f"Correlation: {correlation:.2f}", transform=ax1.transAxes)
        
        # Create bar chart of durations for each question
        # Sorted by duration for better visualization
        sort_by_duration = np.argsort(durations)
        sorted_durations_2 = [durations[i] for i in sort_by_duration]
        sorted_questions = [questions[i] for i in sort_by_duration]
        sorted_difficulties_2 = [difficulties[i] for i in sort_by_duration]
        
        # Create bar chart with color mapped to difficulty
        bars = ax2.barh(range(len(sorted_durations_2)), sorted_durations_2, align='center')
        ax2.set_yticks(range(len(sorted_questions)))
        ax2.set_yticklabels(sorted_questions)
        ax2.set_xlabel('Duration (seconds)')
        ax2.set_title('Evaluation Duration by Question')
        
        # Color bars by difficulty
        norm = plt.Normalize(1, 10)  # 1 is hardest, 10 is easiest
        cmap = plt.cm.get_cmap('RdYlGn')  # Red (hard) to Green (easy)
        
        for i, bar in enumerate(bars):
            bar.set_color(cmap(norm(sorted_difficulties_2[i])))
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Difficulty (10=easiest, 1=hardest)')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        logger.info(f"Plot saved to {output_file}")
        
        # Also create a simple summary statistics file
        stats_file = output_file.replace('.png', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("Duration Statistics (seconds):\n")
            f.write(f"Mean Duration: {np.mean(durations):.2f}\n")
            f.write(f"Median Duration: {np.median(durations):.2f}\n")
            f.write(f"Min Duration: {np.min(durations):.2f}\n")
            f.write(f"Max Duration: {np.max(durations):.2f}\n")
            f.write(f"Standard Deviation: {np.std(durations):.2f}\n\n")
            
            f.write("Correlation between difficulty and duration:\n")
            if len(sorted_difficulties) > 1:
                f.write(f"Correlation Coefficient: {correlation:.2f}\n")
                f.write(f"Linear Fit: Duration = {z[0]:.2f} * Difficulty + {z[1]:.2f}\n")
        
        logger.info(f"Statistics saved to {stats_file}")
        
    except Exception as e:
        logger.error(f"Error plotting durations: {e}")

def plot_duration_correctness_correlation(analysis_file="./src/results/analysis/duration_correctness_analysis.json", 
                                         results_file="./src/results/gsm8k_benchmark_results.json",
                                         output_file="./src/results/analysis/duration_correctness_plot.png"):
    """
    Plot the relationship between duration and correctness based on analysis results.
    
    Args:
        analysis_file: Path to the analysis JSON file
        results_file: Path to the benchmark results file (for scatter plot)
        output_file: Path to save the plot
    """
    try:
        # Load the analysis results
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        # Load the original results for scatter plot
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Top-left: Scatter plot of duration vs correctness
        ax1 = fig.add_subplot(221)
        
        # Extract durations and correctness from original results
        durations = []
        correctness = []
        difficulties = []
        
        for result in results:
            if 'duration' in result and 'numerical_match' in result:
                durations.append(result['duration'])
                correctness.append(1 if result['numerical_match'] else 0)
                difficulties.append(result.get('difficulty', 0))
        
        # Convert to numpy arrays
        durations = np.array(durations)
        correctness = np.array(correctness)
        difficulties = np.array(difficulties)
        
        # Create scatter plot with color by correctness
        colors = ['red' if c == 0 else 'green' for c in correctness]
        ax1.scatter(durations, correctness, c=colors, alpha=0.7)
        
        # Add logistic regression curve if available
        coef = analysis.get('logistic_regression_coefficient')
        intercept = analysis.get('logistic_regression_intercept')
        
        if coef is not None and intercept is not None:
            x_range = np.linspace(min(durations), max(durations), 100)
            y_pred = expit(coef * x_range + intercept)
            ax1.plot(x_range, y_pred, 'b-', linewidth=2)
            
            # Add interpretation text
            if coef > 0:
                direction = "increases"
            else:
                direction = "decreases"
            ax1.text(0.05, 0.9, f"Longer duration {direction} correctness probability", 
                    transform=ax1.transAxes, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Add correlation information
        corr = analysis.get('correlation_coefficient')
        p_val = analysis.get('p_value')
        if corr is not None and p_val is not None:
            significance = "significant" if p_val < 0.05 else "not significant"
            ax1.text(0.05, 0.82, f"Correlation: {corr:.3f} ({significance})", 
                    transform=ax1.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Correctness (0=incorrect, 1=correct)')
        ax1.set_title('Duration vs Correctness Scatter Plot')
        ax1.grid(True, alpha=0.3)
        
        # 2. Top-right: Binned accuracy by duration
        ax2 = fig.add_subplot(222)
        
        bin_centers = analysis.get('binned_durations', [])
        binned_accuracy = analysis.get('binned_accuracy', [])
        
        if bin_centers and binned_accuracy:
            ax2.plot(bin_centers, binned_accuracy, 'o-', linewidth=2)
            ax2.set_xlabel('Duration (seconds)')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy by Duration Bin')
            ax2.grid(True, alpha=0.3)
            
            # Add horizontal line for overall accuracy
            overall_acc = analysis.get('correct_count', 0) / analysis.get('total_count', 1)
            ax2.axhline(y=overall_acc, color='r', linestyle='--', alpha=0.5)
            ax2.text(min(bin_centers), overall_acc + 0.02, f"Overall accuracy: {overall_acc:.2f}", 
                     fontsize=10, color='r')
        else:
            ax2.text(0.5, 0.5, "Not enough data for binned analysis", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes, fontsize=12)
        
        # 3. Bottom-left: Duration comparison (correct vs incorrect)
        ax3 = fig.add_subplot(223)
        
        avg_correct = analysis.get('avg_correct_duration', 0)
        avg_incorrect = analysis.get('avg_incorrect_duration', 0)
        
        bars = ax3.bar(['Incorrect', 'Correct'], [avg_incorrect, avg_correct], 
                       color=['red', 'green'], alpha=0.7)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        ax3.set_ylabel('Average Duration (seconds)')
        ax3.set_title('Average Duration by Correctness')
        
        # 4. Bottom-right: Duration vs difficulty colored by correctness
        ax4 = fig.add_subplot(224)
        
        if len(difficulties) > 0:
            scatter = ax4.scatter(difficulties, durations, c=correctness, 
                                 cmap='RdYlGn', alpha=0.7, s=100)
            
            # Add legend
            handles, labels = scatter.legend_elements()
            legend = ax4.legend(handles, ['Incorrect', 'Correct'], loc="upper right")
            
            # Add trend lines for correct and incorrect answers
            correct_mask = correctness == 1
            incorrect_mask = correctness == 0
            
            if sum(correct_mask) > 1:
                z = np.polyfit(difficulties[correct_mask], durations[correct_mask], 1)
                p = np.poly1d(z)
                ax4.plot(difficulties[correct_mask], p(difficulties[correct_mask]), "g--", alpha=0.5)
            
            if sum(incorrect_mask) > 1:
                z = np.polyfit(difficulties[incorrect_mask], durations[incorrect_mask], 1)
                p = np.poly1d(z)
                ax4.plot(difficulties[incorrect_mask], p(difficulties[incorrect_mask]), "r--", alpha=0.5)
            
            ax4.set_xlabel('Difficulty (10=easiest, 1=hardest)')
            ax4.set_ylabel('Duration (seconds)')
            ax4.set_title('Duration vs Difficulty by Correctness')
            ax4.grid(True, alpha=0.3)
            
            # Add partial correlation information
            partial_corr = analysis.get('partial_correlation_controlling_for_difficulty')
            partial_p = analysis.get('partial_correlation_p_value')
            if partial_corr is not None and partial_p is not None:
                significance = "significant" if partial_p < 0.05 else "not significant"
                ax4.text(0.05, 0.9, f"Partial correlation: {partial_corr:.3f} ({significance})", 
                        transform=ax4.transAxes, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, "No difficulty data available", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax4.transAxes, fontsize=12)
        
        # Add overall title
        plt.suptitle('Relationship Between Duration and Correctness', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for suptitle
        plt.savefig(output_file, dpi=300)
        logger.info(f"Duration-correctness plot saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error plotting duration-correctness correlation: {e}") 