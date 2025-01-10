"""
Simple script to create a list of Paths for each training .csv file in the specified directory.
The list is relevant for running execute_mlde.py with many input files.

"""

# Import dependencies.
import argparse
import re 
from pathlib import Path

# Define function to create input list.
def get_csv_files(directory: Path):
    # List paths to inputs.
    training_list = list(directory.glob('*.csv')) 
    # Sort list by basename number.
    sorted_list = sorted(training_list, key=lambda x: int(re.search(r'(\d+)\.csv$', str(x)).group(1)), reverse=False)
    # Save list to text file.
    with open("training_list.txt", "w") as file:
        for item in sorted_list:
            file.write(f"{item}\n")
        return 

# Define function to parse script arguments.
def parse_arguments():
    parser = argparse.ArgumentParser(description="Returns a text file with a list of paths to each training data input.")
    parser.add_argument("input_directory", type=Path, help="The directory that contains all training .csvs intended for execute_mlde.py.")
    return parser.parse_args()

# Define main.
def main():
    args = parse_arguments()
    get_csv_files(args.input_directory)
    
# Execute if run as a script.
if __name__=="__main__":
    main()