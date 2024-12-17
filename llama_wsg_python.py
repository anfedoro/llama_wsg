#!/opt/homebrew/bin/python3
import argparse
from fastapi import FastAPI, Request
from llama_cpp import Llama
from typing import Dict
import asyncio

# Arguments for script execution
def parse_arguments():
    parser = argparse.ArgumentParser(description="Llama.cpp FastAPI Server with multiple models.")
    parser.add_argument("--models", nargs="+", required=True, help="List of models to load (paths to GGUF files).")
    parser.add_argument("--bind", type=str, default="localhost", help="Bind address for FastAPI.")
    parser.add_argument("--port", type=int, default=8080, help="Port for FastAPI.")
    return parser.parse_args()

args = parse_arguments()

# Initialize FastAPI
app = FastAPI()


# Load models into memory
loaded_models: Dict[str, Llama] = {}
for model_path in args.models:
    print(f"Loading model: {model_path}")
    model_name = model_path.split("/")[-1]  # Get the name of the file
    loaded_models[model_name] = Llama(model_path=model_path, n_ctx=2048)  # Parameters for context size
    model_name = model_path.split("/")[-1]  # Get the name of the file
    loaded_models[model_name] = Llama(model_path=model_path, n_ctx=2048)  # Parameters for context size

print("Models loaded successfully!")

# Main endpoint for handling requests
@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    model_name = body.get("model")
    prompt = body.get("prompt")
    
    if not model_name or model_name not in loaded_models:
        return {"error": "Model not found or not specified."}
    if not prompt:
        return {"error": "Prompt is required."}
    
    model = loaded_models[model_name]
    print(f"Processing request with model: {model_name}")

    # Generate response
    output = model(prompt=prompt, max_tokens=256, stream=False)
    return output

if __name__ == "__main__":
    import uvicorn
    print(f"Starting FastAPI server on {args.bind}:{args.port}")
    uvicorn.run(app, host=args.bind, port=args.port)