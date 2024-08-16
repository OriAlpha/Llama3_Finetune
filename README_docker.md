# Text Generation Inference (TGI) Setup and Usage

Text Generation Inference (TGI) is an optimized framework designed for running and utilizing Large Language Models (LLMs) efficiently. It enhances text generation performance through techniques such as Tensor Parallelism and continuous batching. TGI supports several popular models, including Llama, Mistral, Mixtral and others.

Other than TGI, there are even more OpenSource tools for dockerization of models like BentoML, RayServe or TorchServe.
## Prerequisites

- Docker installed on your system. You can follow the [official Docker installation guide](https://docs.docker.com/engine/install/).
- Ensure sufficient GPU memory is available to run the container.

## Docker Setup

Create a Docker file with the following configuration:

```bash
# Define the model path and configuration
model=model_path  # Set the path to your model
num_shard=1                             # Define the number of shards
max_input_length=1024                   # Set the maximum input length
max_total_tokens=256                    # Set the maximum total tokens

# Run the Docker container
docker run -d --name tgi --gpus all -ti -p 8080:80 \
  -e MODEL_ID=/workspace \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  ghcr.io/huggingface/text-generation-inference:latest
```

## Sending Requests via Python

The following Python code demonstrates how to send a request to the inference server:

```python
import requests as r
from transformers import AutoTokenizer

# Load the tokenizer and evaluation dataset
tokenizer = AutoTokenizer.from_pretrained("model_path")

llama_prompt = """As an assistant for topic modeling, my role is to act considerately, reliably, and as a trustworthy guide in the domain of topic modeling. I aim to provide assistance that meets these standards consistently.

### Input:
{}

### Response:
{}"""

# Replace me to try out new topic
default_topic = "Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a reasonable way (see inductive bias)."

# Generate the prompt using the tokenizer
inputs = tokenizer(
[
    llama_prompt.format(
        default_topic, # input
        "",
    )
], return_tensors = "pt").to("cuda")


request = {
    "inputs": inputs,
    "parameters": {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 256
    }
}

# Send the request to the inference server
resp = r.post("http://127.0.0.1:8080/generate", json=request)

# Extract and print the response
output = resp.json()["generated_text"].strip()


# Display the results
print(f"Output:\n{output}")

```

## Stopping the Docker Container

Once you are finished with the inference, ensure to stop the Docker container to free up resources:

```bash
docker stop tgi
```
