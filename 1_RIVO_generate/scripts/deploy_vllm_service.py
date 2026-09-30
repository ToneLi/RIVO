#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone vLLM OpenAI-compatible server
Usage:
    python deploy_vllm_service.py --model openai/gpt-oss-20b --port 8001 --tensor_parallel_size 1
    python deploy_vllm_service.py --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B --port 8001 --tensor_parallel_size 2
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Start vLLM OpenAI-compatible API server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=None, help="Max model length")
    args = parser.parse_args()

    # Build vLLM command
    cmd_parts = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--enable-prefix-caching",
    ]

    if args.max_model_len:
        cmd_parts.extend(["--max-model-len", str(args.max_model_len)])

    # Print info
    print("=" * 80)
    print("Starting vLLM OpenAI-compatible API server")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print("=" * 80)
    print("\nServer will be available at:")
    print(f"  http://{args.host}:{args.port}/v1")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    print()

    # Execute
    import subprocess
    try:
        subprocess.run(cmd_parts, check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")


if __name__ == "__main__":
    main()
