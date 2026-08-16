import os
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="High-Performance LLM Serving Engine", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration & Hardware-Aware Initialization
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"[Engine] Loading model '{MODEL_NAME}' on {DEVICE} ({DTYPE})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True
)

# Dynamically load LoRA adapter if present
ADAPTER_PATH = "./lora_adapter"
if os.path.exists(ADAPTER_PATH):
    print(f"[Engine] Found trained LoRA adapter at '{ADAPTER_PATH}'. Attaching adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
else:
    print("[Engine] No LoRA adapter found. Running standard base model.")

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    use_kv_cache: bool = True

class GenerationResponse(BaseModel):
    generated_text: str
    latency_ms: float
    tokens_per_second: float

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model": MODEL_NAME
    }

@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    try:
        start_time = time.time()
        
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
        input_length = inputs.input_ids.shape[1]
        
        # Hardware-aware generation loop leveraging explicit KV-Cache
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=True if request.temperature > 0 else False,
                use_cache=request.use_kv_cache,  # Explicit KV Cache flag
                pad_token_id=tokenizer.eos_token_id
            )
        
        latency = (time.time() - start_time) * 1000
        generated_tokens = outputs[0][input_length:]
        gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        num_tokens = len(generated_tokens)
        
        tps = (num_tokens / (latency / 1000.0)) if latency > 0 else 0.0

        return GenerationResponse(
            generated_text=gen_text,
            latency_ms=round(latency, 2),
            tokens_per_second=round(tps, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))