

# Training
`ddp_tranier` and `trainer`script uses for training the model.

arguments: 
The following arguments are available for both trainers:

 - `dataset_dir`: Path to the dataset directory.
 - `batch_size`: Batch size for training (default: 12).
 - `num_epochs`: Number of epochs to train for (default: 30).
 - `learning_rate`: Learning rate for the optimizer (default: 5e-4).
 - `num_warmup_steps`: Number of warmup steps for the learning rate scheduler (default: 500).
 - `save_dir`: Directory to save the best model (default: ./segformer_model).
 - `from_pretrained`: Path to the pretrained model directory.
 - `n_gpus`: Number of GPUs to use (default: 1 for DDP, ignore for standard trainer).
 - `log_name`: Name of the log file (default: segformer_ddp_training.log)

** n_gpus available just in ddp version

<br >

<br >

<br >


# Testing the SegFormer-Lane-Marking Model

for testing clone the preatrained  models from huggingface

```bash
cd <path_to_segformer>
git clone https://huggingface.co/8bits-ai/SegFormer-Lane-Marking models
```


## Script Arguments
The script accepts the following command-line arguments:

--model_path: (Optional) Path to the trained model directory. If not provided, it defaults to the pre-trained SegFormer model from Hugging Face (nvidia/segformer-b2-finetuned-ade-512-512).
--image_path: (Required) Path to the input image file or a directory containing images. The script will process all image files in the directory if a directory is provided.
--gpu: (Optional) GPU device ID to use for inference. Default is 0. If no GPU is available, the script will automatically use the CPU.


## sample usage
```bash
python test.py --model_path models --image_path <path_to_image>
```

## Additional Information

- The output/test directory will be created automatically if it doesn't exist, and all results will be stored there.
- Ensure that your GPU is available and correctly specified with the --gpu argument if you're running the script on a GPU.