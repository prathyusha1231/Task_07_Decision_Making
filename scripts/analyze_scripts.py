"""
NCAA Division 1 Women's Lacrosse 2022-2023 Season Data Analysis

This script performs comprehensive statistical analysis and visualization 
of NCAA Division 1 Women's Lacrosse team performance data.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_and_clean_data(filepath):
    """
    Load and clean the lacrosse dataset
    
    Input: filepath (str) - path to the CSV file
    Output: df (DataFrame) - cleaned pandas DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from string columns
    df['Team'] = df['Team'].str.strip()
    
    return df

def calculate_basic_statistics(df):
    """
    Calculate basic descriptive statistics for all numeric columns
    
    Input: df (DataFrame) - the lacrosse dataset
    Output: stats_dict (dict) - dictionary containing various statistics
    """
    # Select only numeric columns (exclude Team column)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats_dict = {
        'basic_stats': df[numeric_cols].describe(),
        'correlation_matrix': df[numeric_cols].corr(),
        'missing_values': df.isnull().sum(),
        'data_types': df.dtypes
    }
    
    # Additional statistics
    stats_dict['skewness'] = df[numeric_cols].skew()
    stats_dict['kurtosis'] = df[numeric_cols].kurtosis()
    
    return stats_dict

def save_statistics_to_file(stats_dict, df, filename='lacrosse_statistics.txt'):
    """
    Save statistical analysis results to a text file
    
    Input: 
        stats_dict (dict) - dictionary containing statistics
        df (DataFrame) - the original dataset
        filename (str) - output filename
    Output: None
    """
    with open(filename, 'w') as f:
        f.write("NCAA DIVISION 1 WOMEN'S LACROSSE 2022-2023 SEASON ANALYSIS\n")
        f.write("=" * 60 + "\n")

        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total number of teams: {len(df)}\n")
        f.write(f"Number of variables: {len(df.columns)}\n")
        f.write(f"Variables: {', '.join(df.columns)}\n\n")
        
        f.write("BASIC DESCRIPTIVE STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(stats_dict['basic_stats'].to_string())
        f.write("\n\n")
        
        f.write("MISSING VALUES\n")
        f.write("-" * 15 + "\n")
        f.write(stats_dict['missing_values'].to_string())
        f.write("\n\n")
        
        f.write("DATA TYPES\n")
        f.write("-" * 10 + "\n")
        f.write(stats_dict['data_types'].to_string())
        f.write("\n\n")
        
        f.write("SKEWNESS\n")
        f.write("-" * 8 + "\n")
        f.write(stats_dict['skewness'].to_string())
        f.write("\n\n")
        
        f.write("KURTOSIS\n")
        f.write("-" * 8 + "\n")
        f.write(stats_dict['kurtosis'].to_string())
        f.write("\n\n")
        
        # Top performers analysis
        f.write("TOP PERFORMERS\n")
        f.write("-" * 13 + "\n")
        f.write(f"Highest Win Percentage: {df.loc[df['win_pctg'].idxmax(), 'Team']} ({df['win_pctg'].max():.3f})\n")
        f.write(f"Highest Goals Per Game: {df.loc[df['goals_per_game'].idxmax(), 'Team']} ({df['goals_per_game'].max():.2f})\n")
        f.write(f"Best Shot Percentage: {df.loc[df['shot_pctg'].idxmax(), 'Team']} ({df['shot_pctg'].max():.3f})\n")
        f.write(f"Best Save Percentage: {df.loc[df['save_pctg'].idxmax(), 'Team']} ({df['save_pctg'].max():.3f})\n")
        f.write(f"Fewest Goals Allowed: {df.loc[df['goals_allowed_per_game'].idxmin(), 'Team']} ({df['goals_allowed_per_game'].min():.2f})\n\n")
        
        f.write("CORRELATION MATRIX\n")
        f.write("-" * 17 + "\n")
        f.write(stats_dict['correlation_matrix'].to_string())

def create_visualizations(df, output_dir='images'):
    """
    Create various plots for data visualization
    
    Input: 
        df (DataFrame) - the lacrosse dataset
        output_dir (str) - directory to save plots
    Output: None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style for better-looking plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Distribution of Win Percentage
    plt.figure(figsize=(10, 6))
    plt.hist(df['win_pctg'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Win Percentage', fontsize=14, fontweight='bold')
    plt.xlabel('Win Percentage')
    plt.ylabel('Number of Teams')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/win_percentage_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Goals per Game vs Goals Allowed per Game
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['goals_allowed_per_game'], df['goals_per_game'], 
                         c=df['win_pctg'], cmap='RdYlGn', alpha=0.7, s=60)
    plt.colorbar(scatter, label='Win Percentage')
    plt.title('Goals Scored vs Goals Allowed (colored by Win %)', fontsize=14, fontweight='bold')
    plt.xlabel('Goals Allowed per Game')
    plt.ylabel('Goals per Game')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/goals_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(14, 12))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Performance Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Top 10 Teams by Win Percentage
    plt.figure(figsize=(12, 8))
    top_teams = df.nlargest(10, 'win_pctg')
    plt.barh(range(len(top_teams)), top_teams['win_pctg'], color='lightcoral')
    plt.yticks(range(len(top_teams)), [team.split('(')[0].strip() for team in top_teams['Team']])
    plt.title('Top 10 Teams by Win Percentage', fontsize=14, fontweight='bold')
    plt.xlabel('Win Percentage')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_teams_win_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Shot Efficiency Analysis
    plt.figure(figsize=(10, 8))
    plt.scatter(df['shots_per_game'], df['shot_pctg'], 
               c=df['goals_per_game'], cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(label='Goals per Game')
    plt.title('Shot Volume vs Shot Accuracy (colored by Goals per Game)', fontsize=14, fontweight='bold')
    plt.xlabel('Shots per Game')
    plt.ylabel('Shot Percentage')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shot_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Defensive Performance
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['caused_turnovers_per_game'], df['goals_allowed_per_game'], 
               alpha=0.7, color='orange')
    plt.title('Turnovers Caused vs Goals Allowed')
    plt.xlabel('Caused Turnovers per Game')
    plt.ylabel('Goals Allowed per Game')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['save_pctg'], df['goals_allowed_per_game'], 
               alpha=0.7, color='purple')
    plt.title('Save Percentage vs Goals Allowed')
    plt.xlabel('Save Percentage')
    plt.ylabel('Goals Allowed per Game')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Defensive Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/defensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All plots saved to '{output_dir}' directory")

def perform_advanced_analysis(df):
    """
    Perform additional advanced analysis
    
    Input: df (DataFrame) - the lacrosse dataset
    Output: insights (dict) - dictionary with analysis insights
    """
    insights = {}
    
    # Select only numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_df = df[numeric_cols]
    
    # Win percentage correlation analysis
    win_corr = numeric_df.corr()['win_pctg'].abs().sort_values(ascending=False)
    insights['win_correlations'] = win_corr
    
    # Efficiency metrics
    df['offensive_efficiency'] = df['goals_per_game'] / df['shots_per_game']
    df['defensive_efficiency'] = 1 - (df['goals_allowed_per_game'] / df['sog_per_game'])
    
    insights['offensive_efficiency'] = df['offensive_efficiency'].describe()
    insights['defensive_efficiency'] = df['defensive_efficiency'].describe()
    
    return insights

def main():
    """
    Main function to execute the complete analysis pipeline
    
    Input: None
    Output: None
    """
    # Load the data
    print("Loading lacrosse dataset...")
    df = load_and_clean_data('lacrosse_women_ncaa_div1_2022_2023.csv')
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Calculate basic statistics
    print("Calculating basic statistics...")
    stats = calculate_basic_statistics(df)
    
    # Save statistics to file
    print("Saving statistics to file...")
    save_statistics_to_file(stats, df)
    print("Statistics saved to 'lacrosse_statistics.txt'")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df)
    
    # Perform advanced analysis
    print("Performing advanced analysis...")
    insights = perform_advanced_analysis(df)
    
    # Display key insights
    print("\nKEY INSIGHTS:")
    print("-" * 20)
    print(f"Total teams analyzed: {len(df)}")
    print(f"Average win percentage: {df['win_pctg'].mean():.3f}")
    print(f"Average goals per game: {df['goals_per_game'].mean():.2f}")
    print(f"Average goals allowed per game: {df['goals_allowed_per_game'].mean():.2f}")
    
    print("\nTop 3 factors correlated with winning:")
    win_factors = insights['win_correlations'].drop('win_pctg').head(3)
    for factor, correlation in win_factors.items():
        print(f"  {factor}: {correlation:.3f}")
    
    print("\nAnalysis complete! Check 'lacrosse_statistics.txt' and 'images/' folder for results.")

if __name__ == "__main__":
    main()