# SDXL LoRA Pipeline

Goal: Create a pipeline to create and use LoRAs with SDXL for image to image generation.

## Setup
- Create a conda environment
```
conda env create -f environment.yml
```

## Training
- create a directory with all images that shoud be used for training
- create a python script with the following contents:
  ```python
  from lora_trainer import LoRATrainer

  # create a trainer instance
  trainer = LoRATrainer()
  
  # set some training parameters
  trainer.num_train_epochs = 30
  trainer.learning_rate = 1e-04
  trainer.checkpointing_steps = 40
  
  # start training (images and metadata in data/pepper/)
  trainer.train("/path/to/your/dataset/")```
- by default, the trainer will ask for a prompt to train the images with
- optionally, edit the generated metadata.csv to include details for each image

## Generation
```python
from lora_generator import LoRAGenerator

# create instance
generator = LoRAGenerator()

# load your lora
generator.load_lora("loras/your_lora")

# generate image
img = generator.text2img("prompt").images[0]
```
