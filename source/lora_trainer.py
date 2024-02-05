import sys
sys.path.append('diffusers/examples/text_to_image')

from train_text_to_image_lora_sdxl import main as _train_lora, parse_args as _parse_args

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import os
from create_metadata import create_metadata

@dataclass
class LoRATrainer:

    # training parameters
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_vae_model_name_or_path: str = "madebyollin/sdxl-vae-fp16-fix"

    train_data_dir: str | None = None
    caption_column: str = "text"
    resolution: int = 1024
    train_batch_size: int = 1
    num_train_epochs: int = 1
    checkpointing_steps: int = 64
    learning_rate: float = 1e-04
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    mixed_precision: str = "fp16"
    validation_epochs: int = 20
    seed: int = 42
    output_dir: str = "loras"
    validation_prompt: str | None = None
    rank: int = 4
    max_train_steps: int | None = None


    # LoRATrainer parameters
    # image types to look for in the input directory
    image_types: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png"])
    image_label: str | None = None
    # should the user be prompted for inputs
    prompt_user: bool = True


    def _get_input_dir(self, input_dir: str | None) -> str | None:
        '''Get input directory for training. If not provided, use the default train_data_dir.'''

        if input_dir is None:
            input_dir = self.train_data_dir

        if not input_dir or not os.path.exists(input_dir):
            raise ValueError(f"Input directory {input_dir} does not exist.")
        
        return input_dir
    

    def _get_output_dir(self, input_dir: str) -> str:
        '''Get output directory for training. If not provided, create a new directory based on the input directory.'''
        output_dir = self.output_dir
        if output_dir == "loras":
            last_char = input_dir[-1]
            if last_char == "/" or last_char == "\\":
                input_name = os.path.basename(os.path.dirname(input_dir))
            else:
                input_name = os.path.basename(input_dir)
            if not input_name or input_name == "/" or input_name == "." or input_name == "":
                input_name = 'new_lora'
            exists = True
            i = 0
            i_str = ""
            while exists:
                output_dir = os.path.join("loras", f"{input_name}{i_str}")
                exists = os.path.exists(output_dir)
                if exists:
                    i += 1
                    i_str = f"_{i}"
                
        # create output directory
        os.makedirs(output_dir, exist_ok=True)

        return output_dir
    
    
    def _generate_metadata(self, input_dir: str, image_label: str | None) -> bool:
        '''Generate a metadata.csv file for the input directory.
        '''

        # check if image_label is provided
        if image_label is None:
            if self.image_label is not None:
                image_label = self.image_label
            elif self.prompt_user:
                image_label = input("Enter the prompt to train the model: ")
            else:
                print("Failed to generate metadata file. No labels provided.")
                return False
            

        # create metadata.csv file with image paths and labels
        create_metadata(input_dir, image_label)
        print(f"Metadata file created: {os.path.join(input_dir, 'metadata.csv')}")

        # wait to give user opportunity to check and edit the metadata file
        if self.prompt_user:
            input("Edit the metadata file if necessary and press Enter to continue...")

        return True

    
    def _check_metadata(self, input_dir: str, image_label: str | None) -> bool:
        '''Check metadata of input directory and ask user for label column if not provided.
        '''

        # check if a metadata.csv file exists in the input directory
        meta_file = os.path.join(input_dir, "metadata.csv")
        meta_exists = os.path.exists(meta_file)

        # if no metadata file exists, try to generate one
        if not meta_exists:
            generated = self._generate_metadata(input_dir, image_label)
            if not generated:
                print("Failed to locate or generate metadata file.")
                print("Please provide a valid metadata.csv file with file_name and text columns or provide a label.")
                return False


        return True
    

    def _generate_args(self, kv_args: Dict[str, Any]) -> List[str]:
        '''Generate a list of arguments for the training script.
        '''
        args = []
        for k, v in kv_args.items():
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k}")
            elif isinstance(v, (list, tuple)):
                for i in v:
                    args.append(f"--{k}={i}")
            else:
                args.append(f"--{k}={v}")
        
        return args
    
    def _start_training(self, args_list: List[str]):
        '''Start training the LoRA with the given arguments.
        '''
        namespace = _parse_args(args_list)
        _train_lora(namespace)

    def train(self, input_dir: str | None = None, image_label: str | None = None):
        '''Collect training parameters and start training.
        '''

        # check if valid input directory is provided
        input_dir = self._get_input_dir(input_dir)

        # check metadata of input directory and ask user for label column
        # return if user cancels and wants to label the images manually
        cont = self._check_metadata(input_dir, image_label)
        if not cont:
            print("Training cancelled.")
            return
        
        # get output directory
        output_dir = self._get_output_dir(input_dir)

        # generate arguments for the training script
        kv_args = self.__dict__
        if "prompt_user" in kv_args:
            del kv_args["prompt_user"]
        if "image_types" in kv_args:
            del kv_args["image_types"]
        if "image_label" in kv_args:
            del kv_args["image_label"]

        kv_args['train_data_dir'] = input_dir
        kv_args['output_dir'] = output_dir

        # remove None values
        kv_args = {k: v for k, v in kv_args.items() if v is not None}

        args_list = self._generate_args(kv_args)

        print(f"Starting training with: {input_dir}...")
        self._start_training(args_list)
        print(f"Training complete. Model saved to {output_dir}")


