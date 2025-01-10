"""
Simple script to move pdb files from lightdock output folders to a collective folder.
"""

# Import dependencies.
import argparse
from pathlib import Path
import os
import glob
import shutil

def mv_esm_structures(input_dir:Path, output_dir:Path):

    # Create a list of pdb files and identify their path locations
    list_of_pdbs = glob.glob(os.path.join(input_dir, '**', 'current_structure**.pdb'), recursive=True)
    
    # Move files to new folder
    try:
        for pdb in list_of_pdbs:
            shutil.copy2(pdb, output_dir)
    except Exception as e:
        print(f'Error with {pdb}:', e)

# Define function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter top scoring swarm files for each input sequence.")
    parser.add_argument("input_directory", type=Path, help="Path to LightDock output folders.")
    parser.add_argument("output_directory", type=Path, help="Path for the output file to be place.")
    return parser.parse_args()

# Define main.
def main():
    args = parse_arguments()
    mv_esm_structures(args.input_directory, args.output_directory)

# Execute if run as a script.
if __name__=="__main__":
    main()