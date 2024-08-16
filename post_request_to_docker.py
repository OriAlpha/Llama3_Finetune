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