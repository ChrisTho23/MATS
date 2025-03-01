import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set up better styling for plots
plt.style.use('ggplot')
sns.set_palette("viridis")

def load_results(filepath):
    """Load benchmark results from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_temperature_durations(low_temp_file, high_temp_file, output_dir):
    """Compare durations between low and high temperature settings"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the results
    low_temp_results = load_results(low_temp_file)
    high_temp_results = load_results(high_temp_file)
    
    # Ensure the results have the same number of questions
    if len(low_temp_results) != len(high_temp_results):
        print(f"Warning: Different number of questions in files ({len(low_temp_results)} vs {len(high_temp_results)})")
    
    # Extract durations, correctness, and questions
    low_temp_data = [(r.get('question', '')[:30]+'...', 
                      r.get('duration', 0), 
                      r.get('numerical_match', False),
                      r.get('difficulty', 5)) for r in low_temp_results]
    
    high_temp_data = [(r.get('question', '')[:30]+'...', 
                       r.get('duration', 0), 
                       r.get('numerical_match', False),
                       r.get('difficulty', 5)) for r in high_temp_results]
    
    # Create DataFrames
    low_temp_df = pd.DataFrame(low_temp_data, columns=['question', 'duration', 'correct', 'difficulty'])
    high_temp_df = pd.DataFrame(high_temp_data, columns=['question', 'duration', 'correct', 'difficulty'])
    
    # Add temperature column
    low_temp_df['temperature'] = 'Low (0.1)'
    high_temp_df['temperature'] = 'High (0.6)'
    
    # Combine the data
    combined_df = pd.concat([low_temp_df, high_temp_df])
    
    # 1. Create a comparison plot for each question
    plt.figure(figsize=(12, 8))
    
    # Sort for better visualization
    questions = sorted(combined_df['question'].unique())
    
    # Prepare data for side-by-side bars
    df_pivot = combined_df.pivot(index='question', columns='temperature', values='duration').reset_index()
    df_pivot = df_pivot.sort_values(by='Low (0.1)', ascending=False)  # Sort by low temperature duration
    
    x = np.arange(len(df_pivot['question']))
    width = 0.35
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(16, 10))
    rects1 = ax.bar(x - width/2, df_pivot['Low (0.1)'], width, label='Low Temp (0.1)', color='cornflowerblue')
    rects2 = ax.bar(x + width/2, df_pivot['High (0.6)'], width, label='High Temp (0.6)', color='firebrick')
    
    # Add labels and title
    ax.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax.set_title('Generation Time by Question for Different Temperature Settings', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df_pivot['question'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    
    # Add a grid for easier reading
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    add_labels(rects1)
    add_labels(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_duration_by_question.png'), dpi=300)
    
    # 2. Create histograms of duration distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(data=combined_df, x='duration', hue='temperature', element='step', 
                 bins=20, kde=True, common_norm=False, alpha=0.6)
    plt.title('Distribution of Generation Times by Temperature Setting', fontsize=14)
    plt.xlabel('Generation Time (seconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Temperature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_duration_distribution.png'), dpi=300)
    
    # 3. Correct vs Incorrect Analysis
    plt.figure(figsize=(10, 6))
    sns.barplot(data=combined_df, x='temperature', y='duration', hue='correct')
    plt.title('Average Generation Time by Temperature and Correctness', fontsize=14)
    plt.xlabel('Temperature Setting', fontsize=12)
    plt.ylabel('Average Generation Time (seconds)', fontsize=12)
    plt.legend(title='Correct Answer')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_correctness_duration.png'), dpi=300)
    
    # 4. Boxplot comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_df, x='temperature', y='duration')
    plt.title('Distribution of Generation Times by Temperature Setting (Box Plot)', fontsize=14)
    plt.xlabel('Temperature Setting', fontsize=12)
    plt.ylabel('Generation Time (seconds)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_duration_boxplot.png'), dpi=300)
    
    # 5. Scatterplot with difficulty
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=combined_df, x='difficulty', y='duration', hue='temperature', 
                    size='correct', sizes=(100, 200), alpha=0.7)
    plt.title('Generation Time vs. Difficulty by Temperature Setting', fontsize=14)
    plt.xlabel('Problem Difficulty (10 = Easiest, 1 = Hardest)', fontsize=12)
    plt.ylabel('Generation Time (seconds)', fontsize=12)
    
    # Add regression lines
    for temp in ['Low (0.1)', 'High (0.6)']:
        temp_data = combined_df[combined_df['temperature'] == temp]
        if len(temp_data) > 1:
            x = temp_data['difficulty']
            y = temp_data['duration']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), linestyle='--', 
                     label=f'{temp} trend (r={np.corrcoef(x, y)[0,1]:.2f})')
    
    plt.legend(title='Temperature & Correctness')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_difficulty_duration.png'), dpi=300)
    
    # 6. Calculate and print statistics
    print("\n=== Generation Time Statistics ===")
    
    for temp, df in [('Low (0.1)', low_temp_df), ('High (0.6)', high_temp_df)]:
        avg_time = df['duration'].mean()
        median_time = df['duration'].median()
        max_time = df['duration'].max()
        min_time = df['duration'].min()
        std_time = df['duration'].std()
        
        avg_correct = df[df['correct']]['duration'].mean() if not df[df['correct']].empty else 0
        avg_incorrect = df[~df['correct']]['duration'].mean() if not df[~df['correct']].empty else 0
        
        print(f"\nTemperature: {temp}")
        print(f"Average Generation Time: {avg_time:.2f} seconds")
        print(f"Median Generation Time: {median_time:.2f} seconds")
        print(f"Max Generation Time: {max_time:.2f} seconds")
        print(f"Min Generation Time: {min_time:.2f} seconds")
        print(f"Standard Deviation: {std_time:.2f} seconds")
        print(f"Average Time for Correct Answers: {avg_correct:.2f} seconds")
        print(f"Average Time for Incorrect Answers: {avg_incorrect:.2f} seconds")
        
        # Calculate accuracy
        accuracy = df['correct'].mean() * 100
        print(f"Accuracy: {accuracy:.1f}%")
    
    # Save processed data to CSV for further analysis
    combined_df.to_csv(os.path.join(output_dir, 'temperature_duration_data.csv'), index=False)
    
    print(f"\nVisualization files saved to {output_dir}")

if __name__ == "__main__":
    # File paths
    low_temp_file = "./src/results/baseline/low_temp/gsm8k_benchmark_results.json"
    high_temp_file = "./src/results/baseline/high_temp/gsm8k_benchmark_results.json"
    output_dir = "./src/results/"
    
    compare_temperature_durations(low_temp_file, high_temp_file, output_dir) 