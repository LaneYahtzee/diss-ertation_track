"""
Script to sort through all 'rank_filtered.list' files from multiple lightdock simulations and compile all results into a single file.
Also calculates the average scores after averaging the top number of glowworms and min/max normalizes the returned values.
"""

# Import dependencies
import pandas as pd
import argparse
import os
import glob
from pathlib import Path
from itertools import chain

# Define function to identify ID number from directory name
def path_split_search(input_dir:Path, pattern: str):
    """
    Splits the basename of each structure directory to determine the structure ID number. ("structure_567" -> 567)
    """
    # List each section of a path
    split_path_list = list(Path(input_dir).parts)
    # Split pattern into prefix and suffix sections
    prefix, suffix = pattern.split('*', 1)
    # Search each section of the path for the prefix and suffix
    for path_element in split_path_list:
        if path_element.startswith(prefix) and path_element.endswith(suffix):
            return path_element
    return None

# Define function to compile LightDock output values
def search_for_filtered_swarm_list(input_dir:Path, output_dir:Path, file_name:str):
    """
    Function that compiles relevant glowworm data from LightDock filtered, ranked list files.

    Parameters: 
    Input directory (Path): path to the directory that contains all necessary files. Also searches sub-directories, so plan accordingly.
    Output directory (Path): path to the directory where you wish the output file to be generated.
    File Name (str): The name you want the output file to use.

    Returns:
    One .csv file that contains the top glowworm, the average of the top # of glowworms (specified by avg_top_groups object), and the total
    number of glowworms for each structure folder within the input directory. Values for average of the top # of glowworms are min/max normalized.
    """
    # Create a list of lightdock swarm outputs and identify their path locations
    list_of_filtered_ranked_swarms = glob.glob(os.path.join(input_dir, '**', 'filtered', 'rank_filtered.list'), recursive=True)
    
    # Create empty dataframes to contain our compiled values
    top_glowworm_df = pd.DataFrame()
    raw_data_df = pd.DataFrame()

    # Loop over each file from the list of lightdock swarm outputs and compile relevant data
    for file in list_of_filtered_ranked_swarms:
        try:
            df = pd.read_table(file, header=None, delimiter=r'\s+')
            
            # Pull structure # from file path
            path_list = str(path_split_search(file, 'structure_*'))
            path_list = path_list.split('_')
            path_list = path_list[1]
            
            # Pull the raw data and store in a separate dataframe
            raw = df.iloc[:,1].copy()
            raw_name = f'{path_list}'
            raw.name = raw_name

            # Concatenate the raw data to the raw data dataframe
            raw_data_df = pd.concat([raw_data_df, raw], axis=1)

            # Pull the top ranked glowworm and rename columns
            top_glowworm = df.iloc[[0],:]
            top_glowworm.columns = ['top_glowworm_ID', 'normalized_top_glowworm_score', 'unknown_value']
            top_glowworm = top_glowworm.drop(columns='unknown_value')
            top_glowworm.insert(loc=0, column='structure_#', value=path_list)

            # Create a list for the average number of top glowworms to output
            avg_top_groups = [3, 5, 10, 25, 50, 100, 200, 300, 400, 500]

            # Calculate the average docking score for each average number of top glowworms in the list and append it to the dataframe 
            for i, number in enumerate(avg_top_groups):
                if len(df) < number: 
                    top_glowworm.insert(loc=3 + i, column=f'average_score_of_top_{number}_glowworms', value=None)
                else:
                    avg_glowworm = df.iloc[0:number, [1]].mean(axis=0)
                    top_glowworm.insert(loc=3 + i, column=f'average_score_of_top_{number}_glowworms', value=avg_glowworm.iloc[0])

            # Calculate the average docking score using all glowworms in the filtered list
            avg_all_glowworms = df.iloc[:, [1]].mean(axis=0)
            top_glowworm.insert(loc=top_glowworm.shape[1], column=f'average_score_of_all_glowworms', value=avg_all_glowworms.iloc[0])

            # Add a column with the total # of glowworms in the filtered list 
            total_glowworms = len(df)
            top_glowworm.insert(loc=top_glowworm.shape[1], column=f'total_filtered_glowworms', value=total_glowworms)

            # Concatenate the results from this filtered_list to the full dataframe
            top_glowworm_df = pd.concat([top_glowworm_df, top_glowworm], axis=0, ignore_index=True)
        except Exception as e:
            file_path = os.path.basename(file)
            print(f'Error with {file_path}: {e}')
            continue
    
    # Sort indicies in ascending order
    top_glowworm_df['structure_#'] = top_glowworm_df['structure_#'].astype(int)
    top_glowworm_df = top_glowworm_df.sort_values(by=['structure_#'], ascending=True)

    # Reorder columns into ascending order and transpose
    raw_data_df = raw_data_df.reindex(sorted(raw_data_df.columns, key=lambda x: int(x)), axis=1)
    raw_data_df = raw_data_df.transpose().reset_index()
    
    # Rename first column in raw_data_df
    raw_data_df.rename(columns={'index': 'structure_ID'}, inplace=True)
    
    # Rename the remaining columns in raw_data_df
    gw_names = ['glowworm_{}'.format(i) for i in range(1, len(raw_data_df.columns)+1)]
    for i, col in enumerate(raw_data_df.columns[1:], start=1):
        raw_data_df.rename(columns={col: gw_names[i-1]}, inplace=True)

    # Normalize the score columns around their values using min-max normalization
    end_column = top_glowworm_df.columns.get_loc('average_score_of_all_glowworms')
    columns = top_glowworm_df.columns[2:end_column + 1]
    normalized_df = top_glowworm_df.copy()
    for column in columns:
        min_value = normalized_df[column].min()
        max_value = normalized_df[column].max()
        try:
            normalized_df[column] = ((normalized_df[column] - min_value) / (max_value - min_value))
        except ZeroDivisionError as z:
            # For instances when the min and max values are equal (resulting in a 0 denominator). Min/Max normalization cannot be performed.
            # This should only occur if only 1 or 2 values exist in the column or are identical. Replaces values with 1. 
            normalized_df[column] = normalized_df[column].mask(normalized_df[column] == normalized_df[column].max(), 1)
            print("ZeroDivisionError:", z , f"\nMin and Max values in {column} column are the same. \nMin/Max Normalization is not possible. \nSetting all values to 1.")
            continue

    # Save dataframes to csv
    top_glowworm_df.to_csv(f'{output_dir}/{file_name}_unnormalized.csv', index=False)
    normalized_df.to_csv(f'{output_dir}/{file_name}_normalized.csv', index=False) 
    raw_data_df.to_csv(f'{output_dir}/{file_name}_raw_data.csv', index=False)

# Define function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter top scoring swarm files for each input sequence.")
    parser.add_argument("input_directory", type=Path, help="Path to LightDock output folders.")
    parser.add_argument("output_directory", type=Path, help="Path for the output file to be place.")
    parser.add_argument("file_name", type=str, help="Desired file name.")
    return parser.parse_args()

# Define main
def main():
    args = parse_arguments()
    search_for_filtered_swarm_list(args.input_directory, args.output_directory, args.file_name)

#Execute if run as a script.
if __name__ == "__main__":
    main()