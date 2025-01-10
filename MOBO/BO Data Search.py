import pandas as pd
import os
import glob
from pathlib import Path

provided_dir = r"C:\Users\Lane\Downloads\MOBO PDMS Round 3\Ratios 20-24\Ratio 24"
comp_sheet_name = 'Time sweep - 1'
freq_sheet_name = 'Frequency sweep - 1'

def search_rheometry(provided_dir: Path):
    """
    Function that compiles relevant Axial Force data and Complex Modulus data from compression and frequency sweeping files, respectively. 
    Currently designed to pull only the Axial Force column and calculate Axial Stress for compression values as well as
    pull the Complex Modulus columns for frequency sweeping.

    Rheometry data files should be named with a "C" for compression or "F" for frequency sweeping somewhere in the name.
    Avoid using a "c" or "C" in the names of frequency sweeping files and vice versa for compression files.
    Change the 'comp_sheet_name' and 'freq_sheet_name' objects at the top of the script to match the names of the Excel sheets for each
    respective data file. 
    
    Parameters: 
    Input directory (Path): path to the directory that contains all necessary files. Also searches sub-directories.

    Returns:
    Two .csv files titled "combined_compression_axial_stress.csv" and "combined_frequency.csv" that contain the compression data and complex
    modulus data, respectively, from all files under the input directory. 
    """
    # Create lists of both Compression and Frequency Sweeping Data Files
    compression_list = glob.glob(os.path.join(provided_dir, '**', '*C*.xls'), recursive=True)
    freq_list = glob.glob(os.path.join(provided_dir, '**', '*F*.xls'), recursive=True)

    # Create empty DataFrames for each test
    compression_df = pd.DataFrame()
    freq_df = pd.DataFrame()

    # Loop over each file of Compression data
    for file in compression_list:
        try:
            # Read files from list
            df = pd.read_excel(file, header=1, sheet_name=comp_sheet_name)
            axial_force_column = df['Axial force']
            # Rename columns
            file_name = os.path.splitext(os.path.basename(file))[0]
            compression_df[file_name] = axial_force_column
        except Exception as e:
            file_path = os.path.basename(file)
            print(f'Compression - Error reading {file_path}: {e}. \nPlease check that the sheet name and file name match.')
            continue
    # Drop unit index (Newtons, N) and return average values as new index
    if 0 in compression_df.index:
        compression_df.drop(axis=0, index=0, inplace=True)
        compression_df = compression_df / 0.00005
        compression_df.loc['Average'] = compression_df.mean()
    # Identify C10 columns and shift them to the right by 1. Otherwise the order of data would be C0, C10, C5 due to Python's character-to-character comparison method.
    for column in compression_df.columns:
        if column.endswith("10"):
            i = compression_df.columns.get_loc(column)
            compression_df.insert(i+1, column,  compression_df.pop(column))
    # Save file
    compression_df.to_csv(f'{provided_dir}/combined_compression_axial_stress.csv', index=True)

    # Repeat for Frequency Sweeping Data
    for file in freq_list:
        try:
            df = pd.read_excel(file, header=1, sheet_name=freq_sheet_name)
            axial_force_column = df['Complex modulus']
            unit = axial_force_column.iloc[0]
            # Convert MPa to Pa. Default on the current rheometer is MPa but the necessary value is Pa.
            if unit == 'MPa':
                axial_force_column = axial_force_column * 1000000
                axial_force_column.iloc[0] = 'Pa'
            elif unit == 'Pa':
                continue
            else:
                print(f'WARNING: Frequency Sweeping values may not be correct for {os.path.basename(file)}', f'\nCheck units:   Current unit: {unit}, Expected: MPa or Pa.')
            file_name = os.path.splitext(os.path.basename(file))[0]
            freq_df[file_name] = axial_force_column
        except ValueError as ve:
            file_path = os.path.basename(file)
            print(f'Frequency Sweeping - Error reading {file_path}: {ve}. \nPlease check that the sheet name and file name match.' )
            continue
    if 0 in freq_df.index:
        freq_df.drop(axis=0, index=0, inplace=True)
        freq_df.loc['Average'] = freq_df.mean()
    freq_df.to_csv(f'{provided_dir}/combined_frequency.csv', index=True)

search_rheometry(provided_dir)