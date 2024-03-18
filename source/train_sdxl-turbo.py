import lora_trainer


trainer = lora_trainer.LoRATrainer("stabilityai/sdxl-turbo", None, "apes")

trainer.num_train_epochs = 30
trainer.learning_rate = 1e-04
trainer.checkpointing_steps = 40

trainer.train()