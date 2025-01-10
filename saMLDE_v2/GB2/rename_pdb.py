"""
Simple script to rename pdb files from lightdock output folders starting from a particular count.
"""

# Import dependencies.
import argparse
from pathlib import Path
import os
import glob
import re
import shutil

# Define function to pull current_structure number from file name
def number_extraction(file_name):
    pdb_file = re.search(r'current_structure(\d+)', file_name)
    number = int(pdb_file.group(1))
    return number

# Define function to rename files in numeric order starting from a specified number
def rename_pdb(input_dir:Path, output_dir:Path, count_start:int):

    # Create a list of pdb files and sort them by their end number
    list_of_pdbs = glob.glob(os.path.join(input_dir, '**', 'current_structure**.pdb'), recursive=True)
    list_of_pdbs.sort(key=lambda x: number_extraction(x))

    # Define count start
    count = count_start
    
    # Rename the structures in order based on start number
    for file in list_of_pdbs:  
        # Define new file name
        new_name = f"current_structure{count}.pdb"
        # Join the path with the new file name
        new_path = os.path.join(output_dir, new_name)
        # Copy file to output directory with new name
        shutil.copy2(file, new_path)
        # Increase count
        count += 1      

# Define function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter top scoring swarm files for each input sequence.")
    parser.add_argument("input_directory", type=Path, help="Path to LightDock output folders.")
    parser.add_argument("output_directory", type=Path, help="Path for the output file to be place.")
    parser.add_argument("count_start", type=int, help="Number to start renaming count from.")
    return parser.parse_args()

# Define main.
def main():
    args = parse_arguments()
    rename_pdb(args.input_directory, args.output_directory, args.count_start)

# Execute if run as a script.
if __name__=="__main__":
    main()