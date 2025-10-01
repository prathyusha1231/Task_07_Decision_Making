"""
OpenAI LLM Evaluation Script for NCAA Women's Lacrosse Dataset Analysis

This script evaluates OpenAI GPT models' capabilities in answering natural language 
questions about sports performance data, measuring accuracy, reasoning quality, 
and prompt engineering requirements.

"""

import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_evaluation.log'),
        logging.StreamHandler()
    ]
)

class LLMEvaluator:
    """
    Class to evaluate LLM responses to lacrosse dataset questions
    
    Input: OpenAI API client, dataset path, questions list
    Output: Comprehensive evaluation results and metrics
    """
    
    def __init__(self, api_key: str, dataset_path: str, model: str = "gpt-4"):
        """
        Initialize the LLM evaluator
        
        Input:
            api_key (str): OpenAI API key
            dataset_path (str): Path to the lacrosse CSV file
            model (str): OpenAI model to use (default: gpt-4)
        Output: None
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dataset_path = dataset_path
        self.df = None
        self.results = []
        self.load_and_validate_data()
        
    def load_and_validate_data(self):
        """
        Load and validate the lacrosse dataset
        
        Input: None (uses self.dataset_path)
        Output: None (sets self.df)
        """
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.df.columns = self.df.columns.str.strip()
            self.df['Team'] = self.df['Team'].str.strip()
            logging.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise
    
    def get_dataset_summary(self) -> str:
        """
        Generate a comprehensive dataset summary for LLM context
        
        Input: None
        Output: summary (str) - formatted dataset description
        """
        summary = f"""
NCAA Division 1 Women's Lacrosse 2022-2023 Season Dataset

Dataset contains {len(self.df)} teams with {len(self.df.columns)} variables:
{', '.join(self.df.columns)}

Key Statistics:
- Average win percentage: {self.df['win_pctg'].mean():.3f}
- Average goals per game: {self.df['goals_per_game'].mean():.2f}
- Average goals allowed per game: {self.df['goals_allowed_per_game'].mean():.2f}
- Top team by win percentage: {self.df.loc[self.df['win_pctg'].idxmax(), 'Team']} ({self.df['win_pctg'].max():.3f})

Sample data (first 3 teams):
{self.df.head(3).to_string()}
"""
        return summary
    
    def define_questions(self) -> List[Dict[str, Any]]:
        """
        Define the research questions with expected answers and difficulty levels
        
        Input: None
        Output: questions (List[Dict]) - structured question data
        """
        questions = [
            # Easy Questions
            {
                "id": 1,
                "difficulty": "easy",
                "question": "Which team has the highest win_pctg?",
                "expected_answer": self.df.loc[self.df['win_pctg'].idxmax(), 'Team'],
                "validation_method": "exact_match"
            },
            {
                "id": 2,
                "difficulty": "easy", 
                "question": "Which team leads in shot_pctg?",
                "expected_answer": self.df.loc[self.df['shot_pctg'].idxmax(), 'Team'],
                "validation_method": "exact_match"
            },
            {
                "id": 3,
                "difficulty": "easy",
                "question": "What is the median goals_per_game across teams?",
                "expected_answer": self.df['goals_per_game'].median(),
                "validation_method": "numerical_tolerance"
            },
            {
                "id": 4,
                "difficulty": "easy",
                "question": "Which teams are above the league-average draw_pctg?",
                "expected_answer": self.df[self.df['draw_pctg'] > self.df['draw_pctg'].mean()]['Team'].tolist(),
                "validation_method": "list_comparison"
            },
            
            # Medium Questions
            {
                "id": 5,
                "difficulty": "medium",
                "question": "Among teams with above-average draw_pctg, which allowed the fewest goals_allowed_per_game?",
                "expected_answer": self._get_best_defensive_team_with_good_draws(),
                "validation_method": "exact_match"
            },
            {
                "id": 6,
                "difficulty": "medium",
                "question": "Is win_pctg more correlated with draw_pctg or shot_pctg?",
                "expected_answer": self._compare_correlations(),
                "validation_method": "concept_match"
            },
            {
                "id": 7,
                "difficulty": "medium",
                "question": "Which teams have positive (goals_per_game - goals_allowed_per_game) but below-average shot_pctg?",
                "expected_answer": self._get_teams_positive_diff_low_shooting(),
                "validation_method": "list_comparison"
            },
            
            # Hard Questions
            {
                "id": 8,
                "difficulty": "hard",
                "question": "Fit a simple linear model: win_pctg ~ draw_pctg + shot_pctg + turnovers_per_game. Which feature has the largest standardized effect?",
                "expected_answer": self._get_largest_standardized_effect(),
                "validation_method": "concept_match"
            },
            {
                "id": 9,
                "difficulty": "hard",
                "question": "If a team raises draw_pctg by 5 percentage points (0.05), how much does the model predict win_pctg changes (holding others constant)?",
                "expected_answer": self._predict_draw_pctg_change_impact(),
                "validation_method": "numerical_tolerance"
            },
            {
                "id": 10,
                "difficulty": "hard",
                "question": "Recommend one metric to improve for a 'two-more-wins' goal next season; justify with the model and team's current stats.",
                "expected_answer": "This requires contextual analysis based on correlation strengths",
                "validation_method": "reasoning_quality"
            },
            
            # Complex Questions
            {
                "id": 11,
                "difficulty": "complex",
                "question": "Identify the most 'unlucky' team: Which team has significantly better offensive and defensive statistics than their win percentage would suggest?",
                "expected_answer": self._identify_unlucky_team(),
                "validation_method": "reasoning_quality"
            },
            {
                "id": 12,
                "difficulty": "complex",
                "question": "For a team currently at 0.400 win percentage, should they prioritize improving their worst-performing metric or enhancing their best-performing metric? Provide data-driven reasoning.",
                "expected_answer": "Requires strategic analysis of correlation patterns and improvement potential",
                "validation_method": "reasoning_quality"
            },
            {
                "id": 13,
                "difficulty": "complex",
                "question": "Using team performance metrics, rank the conferences by competitive strength and identify which conference shows the most parity.",
                "expected_answer": self._analyze_conference_strength(),
                "validation_method": "reasoning_quality"
            },
            {
                "id": 14,
                "difficulty": "complex",
                "question": "A coach has limited practice time - should they spend 70% on offense, 70% on defense, or 50/50 split? Use correlation analysis and diminishing returns theory.",
                "expected_answer": "Requires analysis of offensive vs defensive correlation with winning",
                "validation_method": "reasoning_quality"
            },
            {
                "id": 15,
                "difficulty": "complex",
                "question": "Based on the statistical patterns, what minimum thresholds in 3 key metrics would a team need to achieve to have an 80% probability of making playoffs?",
                "expected_answer": self._calculate_playoff_thresholds(),
                "validation_method": "reasoning_quality"
            }
        ]
        
        return questions
    
    def _get_best_defensive_team_with_good_draws(self) -> str:
        """Helper method to find team with best defense among good draw teams"""
        above_avg_draw = self.df[self.df['draw_pctg'] > self.df['draw_pctg'].mean()]
        return above_avg_draw.loc[above_avg_draw['goals_allowed_per_game'].idxmin(), 'Team']
    
    def _compare_correlations(self) -> str:
        """Helper method to compare correlations with win percentage"""
        draw_corr = abs(self.df['win_pctg'].corr(self.df['draw_pctg']))
        shot_corr = abs(self.df['win_pctg'].corr(self.df['shot_pctg']))
        return f"draw_pctg (r={draw_corr:.3f})" if draw_corr > shot_corr else f"shot_pctg (r={shot_corr:.3f})"
    
    def _get_teams_positive_diff_low_shooting(self) -> List[str]:
        """Helper method to find teams with positive goal differential but low shooting"""
        goal_diff = self.df['goals_per_game'] - self.df['goals_allowed_per_game']
        avg_shot_pctg = self.df['shot_pctg'].mean()
        mask = (goal_diff > 0) & (self.df['shot_pctg'] < avg_shot_pctg)
        return self.df[mask]['Team'].tolist()
    
    def _get_largest_standardized_effect(self) -> str:
        """Helper method to calculate standardized regression coefficients"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        features = ['draw_pctg', 'shot_pctg', 'turnovers_per_game']
        X = self.df[features].dropna()
        y = self.df.loc[X.index, 'win_pctg']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        coeffs = dict(zip(features, abs(model.coef_)))
        return max(coeffs, key=coeffs.get)
    
    def _predict_draw_pctg_change_impact(self) -> float:
        """Helper method to predict impact of draw_pctg change"""
        from sklearn.linear_model import LinearRegression
        
        features = ['draw_pctg', 'shot_pctg', 'turnovers_per_game']
        X = self.df[features].dropna()
        y = self.df.loc[X.index, 'win_pctg']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Coefficient for draw_pctg represents change in win_pctg per unit change in draw_pctg
        return model.coef_[0] * 0.05  # 5 percentage points = 0.05
    
    def _identify_unlucky_team(self) -> str:
        """Helper method to identify team with best stats but poor record"""
        # Calculate composite offensive and defensive metrics
        self.df['offensive_rating'] = (self.df['goals_per_game'] + self.df['assists_per_game'] + self.df['shot_pctg']) / 3
        self.df['defensive_rating'] = (self.df['save_pctg'] + self.df['caused_turnovers_per_game'] - self.df['goals_allowed_per_game']) / 3
        self.df['expected_performance'] = (self.df['offensive_rating'] + self.df['defensive_rating']) / 2
        self.df['luck_factor'] = self.df['expected_performance'] - self.df['win_pctg']
        
        return self.df.loc[self.df['luck_factor'].idxmax(), 'Team']
    
    def _analyze_conference_strength(self) -> Dict[str, Any]:
        """Helper method to analyze conference strength"""
        # Extract conference from team name (assuming format "Team (Conference)")
        self.df['conference'] = self.df['Team'].str.extract(r'\(([^)]+)\)')
        
        conf_stats = self.df.groupby('conference').agg({
            'win_pctg': ['mean', 'std', 'count'],
            'goals_per_game': 'mean',
            'goals_allowed_per_game': 'mean'
        }).round(3)
        
        return conf_stats.to_dict()
    
    def _calculate_playoff_thresholds(self) -> Dict[str, float]:
        """Helper method to calculate playoff probability thresholds"""
        # Assume top 20% of teams make playoffs
        playoff_threshold = self.df['win_pctg'].quantile(0.8)
        playoff_teams = self.df[self.df['win_pctg'] >= playoff_threshold]
        
        return {
            'goals_per_game': playoff_teams['goals_per_game'].quantile(0.2),
            'goals_allowed_per_game': playoff_teams['goals_allowed_per_game'].quantile(0.8),
            'shot_pctg': playoff_teams['shot_pctg'].quantile(0.2)
        }
    
    def get_relevant_data_for_question(self, question: str) -> str:
        """
        Get relevant subset of data based on the question to stay within token limits
        
        Input: question (str) - the question being asked
        Output: relevant_data (str) - formatted relevant data subset
        """
        question_lower = question.lower()
        
        # For questions about specific metrics, include only relevant columns
        if 'highest win_pctg' in question_lower:
            cols = ['Team', 'win_pctg']
        elif 'shot_pctg' in question_lower and 'leads' in question_lower:
            cols = ['Team', 'shot_pctg']
        elif 'median goals_per_game' in question_lower:
            cols = ['Team', 'goals_per_game']
        elif 'above' in question_lower and 'draw_pctg' in question_lower:
            cols = ['Team', 'draw_pctg']
        elif 'above-average draw_pctg' in question_lower and 'goals_allowed' in question_lower:
            cols = ['Team', 'draw_pctg', 'goals_allowed_per_game']
        elif 'correlation' in question_lower or 'correlated' in question_lower:
            cols = ['Team', 'win_pctg', 'draw_pctg', 'shot_pctg']
        elif 'positive' in question_lower and 'goals_per_game' in question_lower and 'shot_pctg' in question_lower:
            cols = ['Team', 'goals_per_game', 'goals_allowed_per_game', 'shot_pctg']
        elif 'linear model' in question_lower or 'standardized effect' in question_lower:
            cols = ['Team', 'win_pctg', 'draw_pctg', 'shot_pctg', 'turnovers_per_game']
        elif 'raises draw_pctg' in question_lower:
            cols = ['Team', 'win_pctg', 'draw_pctg', 'shot_pctg', 'turnovers_per_game']
        elif 'metric to improve' in question_lower or 'two-more-wins' in question_lower:
            cols = ['Team', 'win_pctg', 'draw_pctg', 'shot_pctg', 'turnovers_per_game', 'goals_per_game', 'goals_allowed_per_game']
        elif 'unlucky' in question_lower:
            cols = ['Team', 'win_pctg', 'goals_per_game', 'goals_allowed_per_game', 'shot_pctg', 'assists_per_game', 'save_pctg', 'caused_turnovers_per_game']
        elif '0.400 win percentage' in question_lower:
            cols = ['Team', 'win_pctg', 'goals_per_game', 'goals_allowed_per_game', 'shot_pctg', 'draw_pctg']
        elif 'conference' in question_lower:
            cols = ['Team', 'win_pctg', 'goals_per_game', 'goals_allowed_per_game', 'shot_pctg']
        elif 'practice time' in question_lower or 'offense' in question_lower and 'defense' in question_lower:
            cols = ['Team', 'win_pctg', 'goals_per_game', 'goals_allowed_per_game', 'shot_pctg', 'save_pctg', 'assists_per_game', 'caused_turnovers_per_game']
        elif 'playoff' in question_lower or 'threshold' in question_lower:
            cols = ['Team', 'win_pctg', 'goals_per_game', 'goals_allowed_per_game', 'shot_pctg', 'draw_pctg']
        else:
            # Default: include key performance indicators
            cols = ['Team', 'win_pctg', 'goals_per_game', 'goals_allowed_per_game', 'shot_pctg', 'draw_pctg']
        
        # Ensure columns exist in dataset
        available_cols = [col for col in cols if col in self.df.columns]
        
        # Return subset as CSV string
        subset_df = self.df[available_cols]
        return subset_df.to_csv(index=False)

    def query_llm(self, question: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
        """
        Query the OpenAI LLM with a question about the dataset
        
        Input:
            question (str): The question to ask
            context (str): Additional context for the question
            max_retries (int): Maximum number of retry attempts
        Output:
            response_data (Dict): Response content and metadata
        """
        # Get relevant data subset to stay within token limits
        relevant_data = self.get_relevant_data_for_question(question)
        
        prompt = f"""
You are a sports analytics expert analyzing NCAA Division 1 Women's Lacrosse data.

Here is the relevant dataset for this question:
{relevant_data}

Dataset Summary:
- Total teams: {len(self.df)}
- This subset shows the most relevant columns for your question
- Full dataset has 18 performance metrics including: {', '.join(self.df.columns)}

{context}

Question: {question}

Please analyze the data above to answer this question. Provide:
1. Your numerical answer (if applicable)
2. Step-by-step reasoning showing your calculations
3. Any assumptions you made
4. Confidence level (high/medium/low)

Use the actual data provided above to perform any necessary calculations or analysis.
"""
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful sports analytics expert with strong statistical analysis skills."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent responses
                    max_tokens=1000
                )
                
                return {
                    "response": response.choices[0].message.content,
                    "tokens_used": response.usage.total_tokens,
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "attempt": attempt + 1
                }
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        "response": f"ERROR: {str(e)}",
                        "tokens_used": 0,
                        "model": self.model,
                        "timestamp": datetime.now().isoformat(),
                        "attempt": attempt + 1
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def evaluate_response(self, question_data: Dict, llm_response: Dict) -> Dict[str, Any]:
        """
        Evaluate the quality and accuracy of an LLM response
        
        Input:
            question_data (Dict): Question information and expected answer
            llm_response (Dict): LLM response data
        Output:
            evaluation (Dict): Comprehensive evaluation metrics
        """
        evaluation = {
            "question_id": question_data["id"],
            "difficulty": question_data["difficulty"],
            "question": question_data["question"],
            "expected_answer": question_data["expected_answer"],
            "llm_response": llm_response["response"],
            "tokens_used": llm_response["tokens_used"],
            "attempts": llm_response["attempt"],
            "timestamp": llm_response["timestamp"]
        }
        
        # Accuracy evaluation based on validation method
        validation_method = question_data["validation_method"]
        
        if validation_method == "exact_match":
            evaluation["accuracy"] = self._evaluate_exact_match(
                question_data["expected_answer"], 
                llm_response["response"]
            )
        elif validation_method == "numerical_tolerance":
            evaluation["accuracy"] = self._evaluate_numerical_tolerance(
                question_data["expected_answer"], 
                llm_response["response"]
            )
        elif validation_method == "list_comparison":
            evaluation["accuracy"] = self._evaluate_list_comparison(
                question_data["expected_answer"], 
                llm_response["response"]
            )
        elif validation_method in ["concept_match", "reasoning_quality"]:
            evaluation["accuracy"] = self._evaluate_reasoning_quality(
                llm_response["response"]
            )
        
        # Additional quality metrics
        evaluation["response_length"] = len(llm_response["response"])
        evaluation["contains_numerical_answer"] = any(char.isdigit() for char in llm_response["response"])
        evaluation["shows_work"] = "step" in llm_response["response"].lower() or "calculation" in llm_response["response"].lower()
        
        return evaluation
    
    def _evaluate_exact_match(self, expected: str, response: str) -> float:
        """Evaluate exact string matching"""
        return 1.0 if expected.lower() in response.lower() else 0.0
    
    def _evaluate_numerical_tolerance(self, expected: float, response: str, tolerance: float = 0.01) -> float:
        """Evaluate numerical answers with tolerance"""
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            for num_str in numbers:
                num = float(num_str)
                if abs(num - expected) <= tolerance:
                    return 1.0
            return 0.0
        except:
            return 0.0
    
    def _evaluate_list_comparison(self, expected: List[str], response: str) -> float:
        """Evaluate list-based answers"""
        found_count = sum(1 for item in expected if item.lower() in response.lower())
        return found_count / len(expected) if expected else 0.0
    
    def _evaluate_reasoning_quality(self, response: str) -> float:
        """Evaluate reasoning quality for complex questions"""
        quality_indicators = [
            "because", "therefore", "analysis", "correlation", "data shows",
            "statistics", "calculate", "compare", "trend", "pattern"
        ]
        
        score = sum(1 for indicator in quality_indicators if indicator in response.lower())
        return min(score / 5, 1.0)  # Normalize to 0-1 scale
    
    def run_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete LLM evaluation process
        
        Input:
            save_results (bool): Whether to save results to files
        Output:
            summary (Dict): Evaluation summary and statistics
        """
        questions = self.define_questions()
        results = []
        
        logging.info(f"Starting evaluation with {len(questions)} questions using {self.model}")
        
        for i, question_data in enumerate(questions, 1):
            print(f"\n{'='*80}")
            print(f"QUESTION {i}/{len(questions)} ({question_data['difficulty'].upper()} LEVEL)")
            print(f"{'='*80}")
            print(f"Q: {question_data['question']}")
            print(f"Expected Answer: {question_data['expected_answer']}")
            
            logging.info(f"Processing question {i}/{len(questions)}: {question_data['difficulty']} level")
            
            # Query LLM
            print(f"\n⏳ Querying {self.model}...")
            llm_response = self.query_llm(question_data["question"])
            
            # Display LLM response
            print(f"\n🤖 LLM RESPONSE:")
            print("-" * 50)
            print(llm_response["response"])
            print("-" * 50)
            
            # Evaluate response
            evaluation = self.evaluate_response(question_data, llm_response)
            results.append(evaluation)
            
            # Display evaluation
            print(f"\n📊 EVALUATION:")
            print(f"   Accuracy Score: {evaluation['accuracy']:.2f}")
            print(f"   Tokens Used: {evaluation['tokens_used']}")
            print(f"   Attempts: {evaluation['attempts']}")
            print(f"   Shows Work: {'Yes' if evaluation['shows_work'] else 'No'}")
            
            # Log progress
            logging.info(f"Question {i} completed. Accuracy: {evaluation['accuracy']:.2f}")
            
            # Brief pause to respect rate limits
            time.sleep(1)
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(results)
        
        if save_results:
            self._save_results(results, summary)
        
        return summary
    
    def _calculate_summary_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation statistics"""
        df_results = pd.DataFrame(results)
        
        summary = {
            "overall_accuracy": df_results['accuracy'].mean(),
            "total_questions": len(results),
            "total_tokens_used": df_results['tokens_used'].sum(),
            "average_response_length": df_results['response_length'].mean(),
            "by_difficulty": df_results.groupby('difficulty')['accuracy'].agg(['mean', 'count']).to_dict(),
            "first_try_success_rate": (df_results['attempts'] == 1).mean(),
            "reasoning_quality_rate": (df_results['shows_work']).mean(),
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model
        }
        
        return summary
    
    def _save_results(self, results: List[Dict], summary: Dict[str, Any]):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'llm_evaluation_results_{self.model}_{timestamp}.csv', index=False)
        
        # Save summary to text file instead of JSON
        with open(f'llm_evaluation_summary_{self.model}_{timestamp}.txt', 'w') as f:
            f.write("LLM EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {summary['model_used']}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Overall Accuracy: {summary['overall_accuracy']:.1%}\n")
            f.write(f"Total Questions: {summary['total_questions']}\n")
            f.write(f"Total Tokens Used: {summary['total_tokens_used']:,}\n")
            f.write(f"First-Try Success Rate: {summary['first_try_success_rate']:.1%}\n")
            f.write(f"Shows Reasoning Rate: {summary['reasoning_quality_rate']:.1%}\n")
            f.write(f"Average Response Length: {summary['average_response_length']:.0f} characters\n\n")
            
            f.write("ACCURACY BY DIFFICULTY LEVEL\n")
            f.write("-" * 30 + "\n")
            for difficulty, stats in summary['by_difficulty']['mean'].items():
                count = summary['by_difficulty']['count'][difficulty]
                f.write(f"{difficulty.capitalize()}: {stats:.1%} ({count} questions)\n")
            
            f.write("\nDETAILED QUESTION RESULTS\n")
            f.write("-" * 25 + "\n")
            for i, result in enumerate(results, 1):
                f.write(f"\nQuestion {i} ({result['difficulty']}):\n")
                f.write(f"  Question: {result['question']}\n")
                f.write(f"  Accuracy: {result['accuracy']:.2f}\n")
                f.write(f"  Tokens Used: {result['tokens_used']}\n")
                f.write(f"  Attempts: {result['attempts']}\n")
                f.write(f"  Shows Work: {'Yes' if result['shows_work'] else 'No'}\n")
        
        logging.info("Results saved to CSV and TXT files")

def main():
    """
    Main function to execute the LLM evaluation pipeline
    
    Input: None
    Output: None
    """
    # Load API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Initialize evaluator with GPT-4 Turbo for larger context window
    evaluator = LLMEvaluator(
        api_key=api_key,
        dataset_path='lacrosse_women_ncaa_div1_2022_2023.csv',
        model='gpt-4-turbo'  # Has 128k context window vs gpt-4's 8k
    )
    
    # Run evaluation
    print("Starting LLM evaluation...")
    summary = evaluator.run_evaluation(save_results=True)
    
    # Display results
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {summary['model_used']}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.1%}")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Total Tokens Used: {summary['total_tokens_used']:,}")
    print(f"First-Try Success Rate: {summary['first_try_success_rate']:.1%}")
    print(f"Shows Reasoning Rate: {summary['reasoning_quality_rate']:.1%}")
    
    print("\nAccuracy by Difficulty:")
    for difficulty, stats in summary['by_difficulty']['mean'].items():
        count = summary['by_difficulty']['count'][difficulty]
        print(f"  {difficulty.capitalize()}: {stats:.1%} ({count} questions)")
    
    print(f"\nDetailed results saved to CSV and JSON files")
    print("Evaluation complete!")

if __name__ == "__main__":
    main()