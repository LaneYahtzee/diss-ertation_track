"""
Script designed to generate necessary training data files to be fed
into the execute_mlde.py file. 

"""

# Import dependencies
import pandas as pd
import argparse
from pathlib import Path

# Define function to generate training files
def training_gen(file_number: int, sample_size: int, fitness_list: Path, output_directory: Path, output_name: str = 'subset_'):
    """
    Randomly selects a subset of values from a full dataframe and saves the subset as a .csv.
    Loops until the defined number of files is reached.

    Parameters:
    file_number (int): the number of subset files you wish to generate.
    sample_size (int): the number of values to include in each subset file.
    fitness_list (Path): the path to the full dataset. Must have 2 columns with headers of 'AACombo' and 'Fitness'.
    output_directory (Path): the path to the location you wish the files to be saved to.
    output_name (str, optional): a name to prefix each output file name. Defaults to 'subset_'.

    Returns:
    Multiple .csv files that are randomly selected subsets of the full fitness_list. 
    Files are named based on their count from 1 to args.file_number.
    """
    
    # Import dataset.
    dataset = pd.read_csv(fitness_list, header=0)

    # Loop over subset sampling.
    count = 1
    while count <= file_number:
        # Randomly sample subset data from dataset.
        selection = pd.DataFrame.sample(dataset, n=sample_size, replace=False, axis=0)
        # Save subset to csv.
        selection.to_csv(f'{output_directory}/{output_name}{count}.csv', index=False)
        count += 1
    # Print confirmation to confirm count is correct.
    print(f"Training file generation complete. {count - 1} files generated.")

# Define function to parse script arguments.
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generates training data input files for execute_mlde.py.")
    parser.add_argument("file_number", type=int, help="The number of training files to be generated.")
    parser.add_argument("sample_size", type=int, help="The number of desired variants to be included in each training set.")
    parser.add_argument("fitness_list", type=Path, help="Path to full list of variants and their fitness scores.")
    parser.add_argument("output_directory", type=Path, help="Path for the output file to be placed.")
    parser.add_argument("--name", type=str, help="Name to be appended to output file names.",
                        required=False, default='subset_')
    return parser.parse_args()

# Define main.
def main():
    args = parse_arguments()
    training_gen(args.file_number, args.sample_size, args.fitness_list, args.output_directory, args.name)

# Execute if run as a script.
if __name__=="__main__":
    main()