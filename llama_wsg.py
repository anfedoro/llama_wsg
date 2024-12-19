import subprocess
import os
import time
import httpx
import asyncio
import argparse
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Optional
from urllib.parse import urlparse, urlunparse
from contextlib import asynccontextmanager

# Global variables
current_model: Optional[str] = None
llama_process: Optional[subprocess.Popen] = None
last_activity_time = time.time()

# Parsing command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Llama.cpp Gateway Proxy to use with two or more models.")
    parser.add_argument("--models-dir", type=str, help="Path to the directory with GGUF models.")
    parser.add_argument("--llama-server", type=str, default="/opt/homebrew/bin/llama-server", help="Path to llama-server executable.")
    parser.add_argument("--wsg-port", type=int, default=8080, help="Port for the FastAPI Gateway (WSG).")
    parser.add_argument("--wsg-bind", type=str, default="127.0.0.1", help="Bind address for the FastAPI Gateway.")
    parser.add_argument("--llama-bind", type=str, default="127.0.0.1", help="Bind address for the Llama Server.")
    parser.add_argument("--llama-port", type=int, default=8000, help="Port for llama.cpp server.")
    parser.add_argument("--timeout", type=int, default=900, help="Inactivity timeout in seconds (-1 for no timeout).")
    return parser.parse_args()

args = parse_arguments()

# FastAPI app and HTTP client
app = FastAPI()
client = httpx.AsyncClient(timeout=None)  # Disable timeout globally

def start_llama_server(model_name: str):
    """Starts the llama.cpp server with the specified model."""
    global llama_process, current_model, last_activity_time

    if llama_process:
        print("Stopping existing llama.cpp server...")
        llama_process.terminate()
        llama_process.wait()
        time.sleep(1)

    model_path = os.path.join(args.models_dir, model_name + ".gguf")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found in {args.models_dir}")

    print(f"Starting llama.cpp server with model: {model_name}")
    llama_process = subprocess.Popen(
        [args.llama_server, "-m", model_path, "-t", "4", "-ngl", "100", "--no-webui", "--port", str(args.llama_port), "--host", str(args.llama_bind)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    current_model = model_name
    last_activity_time = time.time()
    time.sleep(3)
    

def stop_llama_server():
    """Stops the llama.cpp server."""
    global llama_process, current_model
    if llama_process:
        print("Stopping llama.cpp server due to inactivity...")
        llama_process.terminate()
        llama_process.wait()
        llama_process = None
        current_model = None

def update_url(request_url: str, new_port: int, new_host: str) -> str:
    """Updates the host and the port in the URL while keeping the path and parameters."""
    parsed_url = urlparse(request_url)
    updated_netloc = f"{new_host}:{new_port}"
    return urlunparse(parsed_url._replace(netloc=updated_netloc))



async def stream_response(url, method, headers, content):
    """Streaming response from the llama.cpp server."""
    try:
        async with client.stream(method, url, headers=headers, content=content) as response:
            # Проверка типа содержимого
            content_type = response.headers.get("content-type", "")
            if content_type == "text/event-stream":
                # Отправляем потоковые данные
                async for chunk in response.aiter_bytes():
                    yield chunk
            else:
                # Если не поток, возвращаем обычный JSON
                yield await response.aread()
    except httpx.ReadTimeout:
        print("Error: Read timeout while streaming response.")
        yield b'{"error": "Read timeout occurred"}'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan for starting and stopping background tasks."""
    if args.timeout != -1:
        task = asyncio.create_task(inactivity_watcher())
        print("Lifespan: Background watcher started")
    try:
        yield  # the app is starting here
    finally:
        # Graceful Shutdown: Stopping the llama.cpp server
        print("Graceful shutdown: Stopping llama.cpp server...")
        if llama_process:
            stop_llama_server()
        print("Lifespan: Background watcher stopped")
        if args.timeout != -1:
            task.cancel()

async def inactivity_watcher():
    """Background task to check inactivity timeout."""
    global last_activity_time
    while True:
        await asyncio.sleep(60)
        if llama_process and (args.timeout > 0) and (time.time() - last_activity_time > args.timeout):
            stop_llama_server()

@app.post("/{full_path:path}")
async def proxy_request(request: Request):
    """Proxies requests to the llama.cpp server."""
    global current_model, last_activity_time
    print(f"Received HTTP request: {request.url.path}")

    try:
        body = await request.json()
    except Exception:
        body = None

    model_name = request.headers.get("x-model")
    if not model_name:
        return {"error":"'x-model' HTTP header is required in the request"}

    # Updating the last activity time
    last_activity_time = time.time()

    # Starting the server with a new model if needed
    if model_name != current_model:
        try:
            start_llama_server(model_name)
        except FileNotFoundError as e:
            return {"error": str(e)}

    # Proxying the request
    target_url = update_url(str(request.url), args.llama_port, args.llama_bind)

    # Определяем media_type
    media_type = "application/json"
    if "stream" in body and body["stream"]:
        media_type = "text/event-stream"

    return StreamingResponse(
        stream_response(target_url, request.method, request.headers, await request.body()),
        media_type=media_type,
    )

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Gateway on {args.wsg_bind}:{args.wsg_port}...")
    uvicorn.run(app, host=args.wsg_bind, port=args.wsg_port)
