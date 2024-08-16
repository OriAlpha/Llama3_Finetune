# Llama3 Fine-Tuning for Topic Modeling

## Setup Instructions for local setup

### Create Conda Environment

```bash
conda create -n name python=3.10 -y
```

### Install PyTorch

```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Fine-Tuning Llama3 for Topic Modeling

This guide demonstrates how to fine-tune Llama3 for topic modeling. The process involves:

1. Extracting keywords using the KeyBERT library.
2. Using these keywords to create a dataset for the LLM.
3. Combining the instructions to create an instruction-based dataset.
4. Selecting appropriate hyperparameters for model training.
5. Saving adapter and full model weights.

### Execution

Due to hardware limitations, the fine-tuning model part was executed on Google Colab. The docker part could not be tested for the same reason. However, a complete guide for dockerizing the model via TGI from Hugging Face is included.

## Future Improvements

- Create a larger or more detailed dataset.
- Train the model for a higher percentage of the adapter model.
- Experiment with more hyperparameters for efficient model training.
- Train the model on a new GPU that supports flash attention for faster training methods.