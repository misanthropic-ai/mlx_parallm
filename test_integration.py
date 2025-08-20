#!/usr/bin/env python3
"""Integration test for mlx_parallm RL training with Atropos."""

import subprocess
import time
import sys
import requests
import signal
import os
from typing import List, Tuple

def wait_for_port(port: int, timeout: int = 30) -> bool:
    """Wait for a service to start on a port."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/")
            return True
        except:
            time.sleep(0.5)
    return False

def run_integration_test(use_real_atropos: bool = False) -> bool:
    """Run end-to-end integration test.
    
    Args:
        use_real_atropos: If True, use real Atropos with GSM8K. If False, use mock.
    """
    processes: List[Tuple[str, subprocess.Popen]] = []
    
    try:
        if use_real_atropos:
            # 1. Start Atropos API server
            print("Starting Atropos API server on port 8001...")
            atropos_api = subprocess.Popen(
                ["run-api"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/Users/shannon/Workspace/Nous/atropos"
            )
            processes.append(("Atropos API", atropos_api))
            
            # Wait for Atropos API
            if not wait_for_port(8001):
                print("✗ Atropos API failed to start")
                return False
            print("✓ Atropos API is running")
            
            # 2. Start GSM8K environment
            print("Starting GSM8K environment...")
            gsm8k = subprocess.Popen(
                ["python", "environments/gsm8k_server.py", "serve", 
                 "--openai.base_url", "http://localhost:8000/v1",
                 "--openai.api_key", "dummy",
                 "--env.group_size", "4",
                 "--slurm", "false"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/Users/shannon/Workspace/Nous/atropos"
            )
            processes.append(("GSM8K", gsm8k))
            time.sleep(5)  # Give GSM8K time to register with Atropos
            print("✓ GSM8K environment started")
        
        # 3. Start training (which launches inference server)
        print(f"Starting training with {'real Atropos' if use_real_atropos else 'mock client'}...")
        
        if use_real_atropos:
            trainer_cmd = [
                "uv", "run", "mlx_parallm_train",
                "--model-path", "NousResearch/Hermes-4-Qwen3-14B-1-e3",
                "--atropos-url", "http://localhost:8001",
                "--token-budget", "16384",  # Smaller for testing
                "--steps", "2",
                "--batch-size", "2"
            ]
        else:
            # Use mock client (no atropos-url means use mock)
            trainer_cmd = [
                "uv", "run", "mlx_parallm_train",
                "--model-path", "NousResearch/Hermes-4-Qwen3-14B-1-e3",
                "--steps", "2",
                "--batch-size", "2"
            ]
        
        trainer = subprocess.Popen(
            trainer_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd="/Users/shannon/Workspace/artivus/mlx_parallm"
        )
        processes.append(("Trainer", trainer))
        
        # 4. Monitor output
        print("\nMonitoring training output...")
        print("-" * 60)
        
        training_successful = False
        for line in iter(trainer.stdout.readline, b''):
            decoded_line = line.decode().strip()
            print(decoded_line)
            
            # Check for successful training steps
            if "step=2" in decoded_line and "metrics=" in decoded_line:
                print("-" * 60)
                print("\n✓ Training completed 2 steps successfully!")
                training_successful = True
                break
            
            # Check for errors
            if "error" in decoded_line.lower() or "exception" in decoded_line.lower():
                print(f"\n✗ Error detected: {decoded_line}")
        
        if not training_successful:
            print("\n✗ Training did not complete successfully")
            return False
        
        # 5. Wait for server to be ready and test inference
        time.sleep(2)  # Give server time to settle after training
        print("\nTesting inference endpoint...")
        
        try:
            response = requests.post(
                "http://localhost:8000/v1/completions",
                json={
                    "model": "NousResearch/Hermes-4-Qwen3-14B-1-e3",
                    "prompt": "What is 2+2? Answer:",
                    "max_tokens": 10,
                    "temperature": 0.1
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['text']
                print("✓ Inference still working after training!")
                print(f"  Prompt: 'What is 2+2? Answer:'")
                print(f"  Response: '{generated_text}'")
                return True
            else:
                print(f"✗ Inference failed with status {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Could not connect to inference server: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print("\nCleaning up processes...")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    print(f"  ✓ Terminated {name}")
                except subprocess.TimeoutExpired:
                    proc.kill()
                    print(f"  ✓ Killed {name}")

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Integration test for mlx_parallm RL training")
    parser.add_argument(
        "--real-atropos", 
        action="store_true",
        help="Use real Atropos with GSM8K instead of mock client"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("MLX ParaLLM RL Training Integration Test")
    print("=" * 60)
    print(f"Mode: {'Real Atropos + GSM8K' if args.real_atropos else 'Mock Client'}")
    print("=" * 60 + "\n")
    
    success = run_integration_test(use_real_atropos=args.real_atropos)
    
    print("\n" + "=" * 60)
    if success:
        print("✓ INTEGRATION TEST PASSED")
    else:
        print("✗ INTEGRATION TEST FAILED")
    print("=" * 60)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()