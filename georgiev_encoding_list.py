"""
This script generates encodings for a specific provided list of protein variants. 
It is adapted from Wittmann et al. doi: 10.1016/j.cels.2021.07.008.
"""
# Define main
def main():

    # Import dependencies
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import pickle
    import argparse
    import os

    # Import MLDE dependencies
    from code.encode.georgiev_params import GEORGIEV_PARAMETERS  # type: ignore

    # Define arguments for the script
    def parse_arguments():
        # Create argument parser
        parser = argparse.ArgumentParser(description="Generates georgiev encodings for a provided list of protein sequences.")
        # Add arguments
        parser.add_argument("variant_list", help = ".csv file that contains all input variants that will be encoded. There should be no header in the column.", type = Path)
        parser.add_argument("--output_file_name", help = "File name for output file",
                            required = False, default = 'GEORGIEV_ENCODINGS', type = str)
        parser.add_argument("--output_dir", help = "Output directory to store output files", 
                            required = False, default = os.getcwd(), type = Path)
        parser.add_argument("--docking_scores", help = "Path to the file that contains docking scores. There should be a header to identify different features.", 
                            required = False, default = None, type = Path)
        return parser.parse_args()
    
    # Define function to prepare input file for encoding
    def input_processing(input_file: Path):
        """
        Converts protein sequences into lists of characters (ex. 'PYYP' to ['P','Y','Y','P'])
        and returns a list of lists that is ready for MLDE scripts. Column should not have a header.
        """
        # Read input file
        input_df = pd.read_csv(input_file, header = None).iloc[:,0]
        # Create empty dataframe
        processed_input = pd.Series(dtype=object)
        # Loop over each protein and convert the string to individual characters
        for variant in input_df:
            character_list = pd.Series([list(variant)])
            # Add each character set to a pd.Series
            processed_input = pd.concat([processed_input, character_list], axis=0, ignore_index=True)
        # Convert the pd.Series to a list
        processed_input = processed_input.tolist()
        return processed_input
    
    # Define a function that produces dictionaries linking combo and index in the output encodings (edited from encoding_generator.py)
    def build_combo_dicts(input_file:object):
        """
        Builds dictionaries which link the identity of a combination (e.g. ACTV)
        to that combination's index in an encoding array, and vice versa. Both
        dictionaries are saved to disk.
        """
        # Link combo to index and index to combo
        combo_to_index = {"".join(combo): i for i, combo in enumerate(input_file)}
        index_to_combo = {i: "".join(combo) for i, combo in enumerate(input_file)}
        
        # Save the constructed dictionaries
        with open(os.path.join(args.output_dir, f"{args.output_file_name}_ComboToIndex.pkl"), "wb") as f:
            pickle.dump(combo_to_index, f)
        with open(os.path.join(args.output_dir, f"{args.output_file_name}_IndexToCombo.pkl"), "wb") as f:
            pickle.dump(index_to_combo, f)
    
    # Define georgiev encoding function (edited from encoding_generator.py)
    def generate_georgiev(input_file:object):
        """
        Encodes a given combinatorial space with Georgiev parameters.
        """
        # Determine the length of the input protein
        # Assumes all variants are the same length
        variant_length = len(input_file[0]) 
        # Now build all encodings for every combination
        unnorm_encodings = np.empty([len(input_file), variant_length, 19])
        for i, combo in enumerate(input_file):
            unnorm_encodings[i] = [[georgiev_param[character] for georgiev_param
                                    in GEORGIEV_PARAMETERS] for character in combo]
        return unnorm_encodings
    
    # Define a function to introduce the docking features into the numpy encoding array
    def docking_scores(encodings:np.ndarray, scores_file:Path):
        """
        Takes the georgiev encoding matrix and adds an additional scalar feature as a column 
        to each slice of the matrix. The value corresponds with the docking score for that variant.
        Columns of features should have a header.
        """
        # Read scores file
        scores = pd.read_csv(scores_file, header=None)      
        scores = scores.drop(0).reset_index(drop=True)
        
        # Create matrix dimensional objects
        mx_height = encodings.shape[0]
        mx_width = encodings.shape[1]
        mx_depth = encodings.shape[2]
        
        # Create scores dataframe dimensional objects
        df_width = scores.shape[1]

        # Check that the number of provided scores equals the number of matrix slices
        assert len(scores) == mx_height, "Number of docking scores must match number of protein sequences."
        
        # Create a blank matrix to hold the new scores as scalar indices
        scores_arr = np.empty((mx_height, mx_width, df_width))
        
        # Fill the blank matrix with the scores
        for i in range(mx_height):
            for j in range(df_width):
                scores_arr[i, :, j] = scores.iloc[i, j]

        # Concatenate scores matrix to the georgiev matrix
        georgiev_scores_arr = np.concatenate((encodings, scores_arr), axis=2)
        return georgiev_scores_arr
    
    # Parse arguements
    args = parse_arguments()

    # Process input file
    processed_input = input_processing(args.variant_list)

    # Generate georgiev encodings
    unnormalized_embeddings = generate_georgiev(processed_input)
    
    # Check if docking features are included
    if args.docking_scores == None:
        # Generate combinatorial dictionaries (pickle files)
        build_combo_dicts(processed_input)
        # Save unnormalized array
        unnormalized_savename = os.path.join(args.output_dir, f"{args.output_file_name}_control.npy")
        np.save(unnormalized_savename, unnormalized_embeddings)
    else:
        # Generate georgiev encodings with docking features
        docking_matrix = docking_scores(unnormalized_embeddings, args.docking_scores)
        # Save the modified array
        scores_array_name = os.path.join(args.output_dir, f"{args.output_file_name}.npy")
        np.save(scores_array_name, docking_matrix)

# Execute if run as a script
if __name__=="__main__":
    main()