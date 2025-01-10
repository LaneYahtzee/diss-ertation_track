import argparse
import subprocess
import shutil
from pathlib import Path
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import logging
from typing import List, Optional, Dict
import os
import re
import fnmatch
import yaml
import shutil
import uuid
import glob
from datetime import datetime


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProteinProcessor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict_structure(self, sequence: str) -> Dict:
        if sequence.startswith('\ufeff'):
            sequence = sequence.lstrip('\ufeff')
        inputs = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        return self.model(**inputs)

    @staticmethod
    def convert_esm_to_pdb(outputs: Dict) -> Optional[str]:
        try:
            outputs = {k: v.cpu().detach().numpy() for k, v in outputs.items()}
            atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
            protein = OFProtein(
                aatype=outputs["aatype"][0],
                atom_positions=atom_positions[0],
                atom_mask=outputs["atom37_atom_exists"][0],
                residue_index=outputs["residue_index"][0] + 1,
                b_factors=outputs["plddt"][0],
                chain_index=outputs.get("chain_index", {}).get(0)
            )
            return to_pdb(protein)
        except Exception as e:
            logger.error(f"Error converting ESM output to PDB: {e}")
            return None

class FileManager:
    @staticmethod
    def get_csv_files(directory: Path) -> List[Path]:
        return list(directory.glob('*.csv'))

    @staticmethod
    def read_sequences_from_csv(csv_file: Path) -> List[str]:
        try:
            with open(csv_file, "r") as f:
                return f.read().splitlines()
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_file}: {e}")
            return []

    @staticmethod
    def save_pdb_structure(pdb_structure: str, output_dir: Path, filename: str):
        pdb_file = output_dir / filename
        pdb_file.write_text(pdb_structure)
        logger.info(f"Saved PDB structure to {pdb_file}")

    @staticmethod
    def create_structure_folder(output_dir: Path, folder_name: str) -> Path:
        folder = output_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder
    
    @staticmethod
    def copy_input_dir(input_dir: Path, structure_folder: Path, folder_name: str) -> Path:
        folder = structure_folder / folder_name
        shutil.copytree(input_dir, folder)
        return folder

    @staticmethod
    def remove_lightdock_files(input_dir: Path, reference_pdb: str):
        files_to_remove = [
            input_dir / f"lightdock_{reference_pdb}",
            input_dir / f"lightdock_{reference_pdb.rsplit('.', 1)[0]}_mask.npy"
        ]
        for file in files_to_remove:
            if file.exists():
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
                logger.info(f"Removed existing file/directory: {file}")
        for file in input_dir.glob('lightdock_*'):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
            logger.info(f"Removed additional lightdock file/directory: {file}")


class CommandRunner:
    def __init__(self, config: Dict):
        self.config = config

    def run_command(self, command: str):
        logger.info(f"Executing command: {command}")
        try:
            result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.info(f"Command executed successfully. Output:\n{result.stdout}")
            if "lightdock3_setup" in command:
                logger.info(f"LightDock setup output:\n{result.stdout}\n{result.stderr}")
            elif "lgd_rank" in command:
                logger.info(f"Swarm ranking output:\n{result.stdout}\n{result.stderr}")
            elif "lgd_filter_restraints" in command:
                logger.info(f"Filtered swarm ranks output:\n{result.stdout}\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing command: {command}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Error output:\n{e.stderr}")
            if "lightdock3_setup" in command:
                logger.error(f"LightDock setup error details:\n{e.stdout}\n{e.stderr}")
            elif "lgd_rank" in command:
                logger.info(f"Swarm ranking output:\n{e.stdout}\n{e.stderr}")
            elif "lgd_filter_restraints" in command:
                logger.info(f"Filtered swarm ranks output:\n{e.stdout}\n{e.stderr}")
            raise

    def run_commands_on_swarm_files(self, root_dir: str, folder_pattern: str, file_pattern: str, input_dir: str, output_dir: str, pdb_file: str):
        commands = self.config['swarm_commands']
        for folder in os.listdir(root_dir):
            current_structure_index = str(root_dir)
            current_structure_index = current_structure_index[-1]
            folder_path = os.path.join(root_dir, folder)
            if not (fnmatch.fnmatch(folder, folder_pattern) and os.path.isdir(folder_path)):
                continue
            largest_number = -1
            largest_file = None
            for file in os.listdir(folder_path):
                if not fnmatch.fnmatch(file, file_pattern):
                    continue
                match = re.search(r'(\d+)\.out$', file)
                if match:
                    number = int(match.group(1))
                    if number > largest_number:
                        largest_number = number
                        largest_file = file
            os.chdir(folder_path)    
            self._execute_commands(commands, folder_path, largest_file, input_dir, output_dir, current_structure_index, pdb_file)
    
    def _execute_commands(self, commands: List[str], folder_path: str, largest_file: str, input_dir: str, output_dir: str, current_structure_index: str, pdb_file: str):
        for command in commands:
            formatted_command = command.format(
                input_dir=input_dir,
                output_dir=output_dir,
                largest_file=largest_file,
                current_structure_index=current_structure_index,
                reference_pdb=self.config.get('reference_pdb'),
                pdb_file=pdb_file
            )
            logger.info(f"Executing command in {folder_path}: {formatted_command}")
            self.run_command(formatted_command)
            logger.info(f"Command executed successfully: {formatted_command}")

class ProteinDockingPipeline:
    def __init__(self, config: Dict, input_seq_index: int, input_seq: str, ):
        self.input_dir = Path(config['paths']['input_dir'])
        self.output_dir = Path(config['paths']['output_dir'])
        self.protein_processor = None
        self.file_manager = FileManager()
        self.command_runner = CommandRunner(config)
        self.config = config
        self.input_seq_index = input_seq_index
        self.input_seq = input_seq
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    def initialize_model(self):
        model = EsmForProteinFolding.from_pretrained(
            self.config['model']['name'], 
            cache_dir=self.config['model']['cache_dir']
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'], 
            cache_dir=self.config['model']['cache_dir']
        )
        self.protein_processor = ProteinProcessor(model, tokenizer)

    def process_input_file(self):
        structure_folder = self.file_manager.create_structure_folder(
            self.output_dir, 
            self.config['output_structure_folder'].format(parsed_int=self.input_seq_index)
        )
        input_copy = self.file_manager.copy_input_dir(self.input_dir, structure_folder, 'INPUT')
        self.input_dir = input_copy
        pdb_structure = self._generate_pdb_structure(self.input_seq)
        if pdb_structure:
            output_filename = f"current_structure{self.input_seq_index}.pdb"
            self.file_manager.save_pdb_structure(pdb_structure, structure_folder, output_filename)
            logger.info(f"Processed sequence {self.input_seq_index}")
        else:
            logger.warning(f"Failed to generate PDB structure for sequence {self.input_seq_index}")
        self.run_lightdock_commands(structure_folder)
        self.delete_unnecessary_files(structure_folder)

    def _generate_pdb_structure(self, sequence: str):
        raw_structure = self.protein_processor.predict_structure(sequence)
        return self.protein_processor.convert_esm_to_pdb(raw_structure)

    def generate_ranked_swarm_list(self, structure_folder: Path):
        os.chdir(structure_folder)
        command = self.config['output_file_commands'][0]
        self.command_runner.run_command(command)

    def threshold_ranked_swarm_list(self, structure_folder: Path):
        os.chdir(structure_folder)
        command = self.config['output_file_commands'][1]
        formatted_command = command.format(
                input_dir=self.input_dir,
                structure_folder=structure_folder
            )
        self.command_runner.run_command(formatted_command)

    def run_lightdock_commands(self, structure_folder: Path):
        pdb_files = list(structure_folder.glob('current_structure*.pdb'))
        pdb_file = pdb_files[0]
        parsed_int = pdb_file.stem.split('current_structure')[1]
        self.file_manager.remove_lightdock_files(self.input_dir, self.config.get('reference_pdb'))
        for command in self.config['setup_commands']:
            formatted_command = command.format(
                input_dir=self.input_dir,
                output_dir=structure_folder,
                pdb_file=pdb_file,
                parsed_int=parsed_int,
                reference_pdb=self.config.get('reference_pdb')
            )
            try:
                 os.chdir(structure_folder)
                 self.command_runner.run_command(formatted_command)
            except subprocess.CalledProcessError:
                logger.error(f"Failed to run setup command for {pdb_file}. Skipping to next file.")
                continue
        self.command_runner.run_commands_on_swarm_files(
            structure_folder, 
            self.config['swarm_folder_pattern'],
            self.config['swarm_file_pattern'],
            str(self.input_dir), 
            str(self.output_dir),
            pdb_file=pdb_file
        )
        self.generate_ranked_swarm_list(structure_folder)
        self.threshold_ranked_swarm_list(structure_folder)

    def delete_unnecessary_files(self, structure_folder: Path):
        shutil.rmtree(fr'{structure_folder}/init')
        shutil.rmtree(fr'{self.input_dir}')
        
        swarm_pattern = self.config['swarm_folder_pattern']
        swarm_dir_list = glob.glob(fr'{structure_folder}//{swarm_pattern}')
        lightdock_file_list = glob.glob(fr'{structure_folder}/lightdock_*')
        filter_structures_list = glob.glob(fr'{structure_folder}/filtered/*.pdb')
        for folder in swarm_dir_list:
            shutil.rmtree(folder)
        for file in lightdock_file_list:
            os.remove(file)
        for file in filter_structures_list:
            os.remove(file)
        
def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process protein sequences and PDB files")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("input_seq_index", type=int, help="The index for the input sequence in your input csv file. This is used for tracking multi-sequence jobs")
    parser.add_argument("input_seq", type=str, help="Input sequence for ligand protein")
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = load_config(args.config)
    input_dir = Path(config['paths']['input_dir'])
    output_dir = Path(config['paths']['output_dir'])
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ProteinDockingPipeline(config, args.input_seq_index, args.input_seq)
    pipeline.initialize_model()
    pipeline.process_input_file()

if __name__ == "__main__":
    main()