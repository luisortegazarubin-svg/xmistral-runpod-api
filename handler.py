import runpod
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from huggingface_hub import login

# --- Global variables to hold the model and tokenizer ---
model = None
tokenizer = None

def init():
    """
    This function is called once when the worker starts.
    It now logs into Hugging Face before loading the model.
    """
    global model, tokenizer

    # --- Securely log in to Hugging Face using the RunPod Secret ---
    hf_token = os.getenv("HUGGING_FACE_TOKEN", None)
    if hf_token:
        print("Hugging Face token found, logging in...")
        login(token=hf_token)
    else:
        print("WARNING: No Hugging Face token found. Will only be able to access public models.")
    # --- End of Login Block ---

    print("Loading base model and private adapter...")
    
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    adapter_id = "fauxstar/XMistral" # Your private model on Hugging Face

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load the base model in 4-bit for efficiency
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )

    # Apply your LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model = model.merge_and_unload() # Optimize for faster inference

    print("Setup complete. Model is ready.")
    return model, tokenizer

def handler(job):
    """
    This function is called for every API request.
    The 'job' object contains the input from the user.
    """
    global model, tokenizer
    # Ensure the model is loaded, calling init() if it's the first run
    if model is None or tokenizer is None:
        model, tokenizer = init()

    job_input = job['input']
    
    # Get parameters from the API call, with default values
    prompt = job_input.get('prompt', 'You are a pirate. What do you see?')
    max_new_tokens = job_input.get('max_new_tokens', 512)
    temperature = job_input.get('temperature', 0.7)

    # Format the prompt using the chat template
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    # Generate the output
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    # Return the result in the format RunPod expects
    return {"result": response}

# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
