"""
Script to sort through all 'PredictedFitness.csv' files from multiple mlde simulations, calculate Spearman's correlation for each
file, and then return the scores in a single .csv file.
"""

# Import dependencies
import pandas as pd
import argparse
import os
import glob
import re
from pathlib import Path

# Define function to pull subset ID number from directory name
def path_split_search(input_dir:Path, pattern: str):
    """
    Splits the basename of each subset directory to determine the subset ID number. ("PROTEIN_subset_567" -> 567)
    """
    # List each section of a path
    split_path_list = list(Path(input_dir).parts)
    # Check each section for a pattern
    for path_element in split_path_list:
        if re.search(pattern, path_element):
            return path_element
    return None

# Define function to compile lightdock swarm values for each input variant
def compile_spearmans(true_scores: Path, input_dir:Path, output_dir:Path, file_name:str):
    """
    Calculates Spearman's correlation coefficient for each list of predicted fitness scores vs a reference.

    Parameters: 
    Input directory (Path): path to the directory that contains all necessary files. Also searches sub-directories.
    Output directory (Path): path to the directory where you wish the output file to be generated.
    File Name (str): The name you want the output file to use.

    Returns:
    One .csv file that contains the Spearman's correlation coefficient for each subset training dataset.
    """
    # Load the true fitness scores into a dataframe
    actual_fitness = pd.read_csv(true_scores, header=0)
    
    # Create a list of paths to each PredictedFitness.csv file
    fitness_file_list = glob.glob(os.path.join(input_dir,'**', 'PredictedFitness.csv'), recursive=True)
    
    # Create an empty dataframe to contain our compiled values
    spearman_df = pd.DataFrame(columns=['Subset_ID', 'Spearman\'s Coefficient'])

    # Loop over each file from the list of PredictedFitness.csv paths
    for file in fitness_file_list:
        try:
            # Read PredictedFitness file
            predicted_fitness = pd.read_csv(file, header=0)
            
            # Pull subset # from file path.
            path_list = str(path_split_search(file, r'subset_\d+'))
            path_list = path_list.split('_')
            path_list = path_list[-1]

            # Merge the actual and predicted fitness dataframes
            merged_df = pd.merge(actual_fitness, predicted_fitness, on='AACombo')
            # Calculate Spearman's correlation
            spearman_corr = merged_df[['Fitness', 'PredictedFitness']].corr(method='spearman')
            # Slice correlation matrix to acquire only the correlation between the two
            spearman_corr = spearman_corr.loc['Fitness', 'PredictedFitness']
            
            # Create a dataframe to contain these values
            insert_df = pd.DataFrame({'Subset_ID': [path_list], 'Spearman\'s Coefficient': [spearman_corr]})

            # Concatenate the values to the full dataframe
            spearman_df = pd.concat([spearman_df, insert_df], axis=0, ignore_index=True)

        # Handle exceptions    
        except Exception as e:
            file_path = os.path.basename(file)
            print(f'Error with {file_path}: {e}')
            continue
    
    # Sort values in ascending order and write to csv file.
    spearman_df['Subset_ID'] = spearman_df['Subset_ID'].astype(int)
    spearman_df = spearman_df.sort_values(by=['Subset_ID'], ascending=True)
    spearman_df.to_csv(f'{output_dir}/{file_name}.csv', index=False)  

# Define function to parse arguments.
def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate Spearman's correlation for all mlde training inputs.")
    parser.add_argument("actual_fitness_scores", type=Path, help="Path to .csv file with actual fitness scores.")
    parser.add_argument("MLDE_file_directory", type=Path, help="Path to MLDE output folders.")
    parser.add_argument("output_directory", type=Path, help="Path for the output file to be placed.")
    parser.add_argument("file_name", type=str, help="Desired file name.")
    return parser.parse_args()

# Define main.
def main():
    args = parse_arguments()
    compile_spearmans(args.actual_fitness_scores, args.MLDE_file_directory, args.output_directory, args.file_name)


if __name__ == "__main__":
    main()