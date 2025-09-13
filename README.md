# GPT-2 From Scratch

This project implements GPT-2 from scratch in PyTorch, including model architecture, training, and evaluation. The model is pretrained on the FineWeb dataset for language modeling and downstream tasks. 
This repo domy work following Andrej Karpathy's video

## Features
- Custom GPT-2 implementation (no HuggingFace code)
- Training scripts for large-scale datasets
- Evaluation and visualization tools
- Demo notebook for inference

## Training
The model is trained on the FineWeb dataset using scripts in the `training/` directory. Training logs and checkpoints are saved for analysis and reproducibility.

## Usage
To run the pretrained model and generate text:

1. Open `demo/play.ipynb` in Jupyter or VS Code.
2. Run the cells to load the model and generate samples.


Install dependencies:
```bash
pip install -r requirements.txt
```

## Files
- `model/` — Model code and configuration
- `training/` — Training scripts
- `demo/play.ipynb` — Demo notebook for inference
- `log/` — Training logs and checkpoints
