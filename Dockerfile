# Start from an official RunPod PyTorch image
FROM runpod/pytorch:2.1.2-py3.11-cuda12.1.1-devel-22.04

# Install required Python packages
RUN pip install --upgrade pip && \
    pip install transformers==4.36.2 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.3 huggingface-hub==0.20.3

# Copy your handler file into the container
COPY handler.py /

# Command to start the worker when the container runs
CMD ["python", "-u", "/handler.py"]
```5.  Haz clic en **"Commit changes..."**.
