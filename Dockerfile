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