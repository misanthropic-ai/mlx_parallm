#!/usr/bin/env python3
"""
Launch script for MLX ParaLLM RL training with Atropos.

This script starts all required services in the correct order:
1. Atropos API server
2. GSM8K environment (or other Atropos environment)
3. MLX ParaLLM training (which includes inference server)

Each service runs in a subprocess with output logging.
"""

import argparse
import subprocess
import time
import sys
import os
import signal
from pathlib import Path
from typing import List, Tuple, Optional
import threading
from datetime import datetime
import requests
import atexit

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class ServiceManager:
    """Manages subprocess services with logging."""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.processes: List[Tuple[str, subprocess.Popen]] = []
        self.log_dir = Path(log_dir) if log_dir else Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n{Colors.YELLOW}Received signal {signum}, shutting down...{Colors.ENDC}")
        self.cleanup()
        sys.exit(0)
    
    def log_output(self, name: str, process: subprocess.Popen, stream_name: str):
        """Log subprocess output to file and console."""
        log_file = self.log_dir / f"{name}_{stream_name}.log"
        
        def reader():
            stream = process.stdout if stream_name == "stdout" else process.stderr
            with open(log_file, "w") as f:
                for line in iter(stream.readline, b''):
                    if self.stop_event.is_set():
                        break
                    if line:
                        decoded = line.decode('utf-8', errors='replace')
                        f.write(decoded)
                        f.flush()
                        
                        # Color-code output by service
                        if name == "atropos-api":
                            prefix = f"{Colors.CYAN}[ATROPOS]{Colors.ENDC}"
                        elif name == "gsm8k-env":
                            prefix = f"{Colors.BLUE}[GSM8K]{Colors.ENDC}"
                        elif name == "trainer":
                            prefix = f"{Colors.GREEN}[TRAINER]{Colors.ENDC}"
                        else:
                            prefix = f"[{name.upper()}]"
                        
                        # Highlight errors
                        if "error" in decoded.lower() or "exception" in decoded.lower():
                            print(f"{prefix} {Colors.RED}{decoded.rstrip()}{Colors.ENDC}")
                        elif "warning" in decoded.lower():
                            print(f"{prefix} {Colors.YELLOW}{decoded.rstrip()}{Colors.ENDC}")
                        else:
                            print(f"{prefix} {decoded.rstrip()}")
        
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        return thread
    
    def start_service(self, name: str, cmd: List[str], cwd: Optional[str] = None, 
                      env: Optional[dict] = None) -> subprocess.Popen:
        """Start a service subprocess with logging."""
        print(f"{Colors.BOLD}Starting {name}...{Colors.ENDC}")
        print(f"  Command: {' '.join(cmd)}")
        if cwd:
            print(f"  Working dir: {cwd}")
        
        # Merge environment variables
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=full_env,
            bufsize=1
        )
        
        self.processes.append((name, process))
        
        # Start logging threads
        self.log_output(name, process, "stdout")
        self.log_output(name, process, "stderr")
        
        return process
    
    def wait_for_port(self, port: int, service_name: str, timeout: int = 30) -> bool:
        """Wait for a service to be available on a port."""
        print(f"Waiting for {service_name} on port {port}...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/", timeout=1)
                print(f"{Colors.GREEN}✓ {service_name} is ready on port {port}{Colors.ENDC}")
                return True
            except:
                time.sleep(1)
        
        print(f"{Colors.RED}✗ {service_name} failed to start on port {port}{Colors.ENDC}")
        return False
    
    def cleanup(self):
        """Clean up all processes."""
        self.stop_event.set()
        print(f"\n{Colors.YELLOW}Cleaning up processes...{Colors.ENDC}")
        
        for name, process in self.processes:
            if process.poll() is None:
                print(f"  Terminating {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force killing {name}...")
                    process.kill()
                    process.wait()
        
        print(f"{Colors.GREEN}All processes terminated{Colors.ENDC}")
        print(f"Logs saved to: {self.log_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Launch MLX ParaLLM RL training with Atropos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use mock client for testing (no Atropos needed)
  python launch_rl_training.py --mock
  
  # Use real Atropos with GSM8K
  python launch_rl_training.py --model NousResearch/Hermes-4-Qwen3-14B-1-e3
  
  # Custom configuration
  python launch_rl_training.py \\
    --model NousResearch/Hermes-4-Qwen3-14B-1-e3 \\
    --steps 10 \\
    --batch-size 4 \\
    --group-size 8
        """
    )
    
    # Model configuration
    parser.add_argument("--model", default="NousResearch/Hermes-4-Qwen3-14B-1-e3",
                       help="Model path or HuggingFace ID")
    parser.add_argument("--lora-path", help="Optional LoRA adapter path")
    
    # Training configuration
    parser.add_argument("--steps", type=int, default=5,
                       help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--token-budget", type=int, default=16384,
                       help="Token budget for Atropos batches")
    
    # Atropos configuration
    parser.add_argument("--mock", action="store_true",
                       help="Use mock client instead of real Atropos")
    parser.add_argument("--environment", default="gsm8k",
                       choices=["gsm8k", "math", "letter_counting"],
                       help="Atropos environment to use")
    parser.add_argument("--group-size", type=int, default=4,
                       help="Group size for Atropos environment")
    
    # Server configuration
    parser.add_argument("--inference-port", type=int, default=8000,
                       help="Port for inference server")
    parser.add_argument("--atropos-port", type=int, default=8001,
                       help="Port for Atropos API")
    
    # Other options
    parser.add_argument("--log-dir", help="Directory for log files")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for adapter checkpoints")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Don't clean up processes on exit")
    
    args = parser.parse_args()
    
    # Print configuration
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}MLX ParaLLM RL Training Launcher{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"Model: {args.model}")
    print(f"Mode: {'Mock Client' if args.mock else f'Real Atropos ({args.environment})'}")
    print(f"Steps: {args.steps}")
    print(f"Batch Size: {args.batch_size}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
    
    # Create service manager
    manager = ServiceManager(log_dir=args.log_dir)
    
    try:
        if not args.mock:
            # Start Atropos API server
            atropos_dir = "/Users/shannon/Workspace/Nous/atropos"
            if not Path(atropos_dir).exists():
                print(f"{Colors.RED}Atropos directory not found: {atropos_dir}{Colors.ENDC}")
                sys.exit(1)
            
            manager.start_service(
                "atropos-api",
                ["uv", "run", "run-api", "--port", str(args.atropos_port)],
                cwd=atropos_dir
            )
            
            # Wait for Atropos API
            if not manager.wait_for_port(args.atropos_port, "Atropos API"):
                print(f"{Colors.RED}Failed to start Atropos API{Colors.ENDC}")
                sys.exit(1)
            
            # Start environment server
            env_script = f"environments/{args.environment}_server.py"
            manager.start_service(
                f"{args.environment}-env",
                [
                    "uv", "run", "python", env_script, "serve",
                    "--openai.base_url", f"http://localhost:{args.inference_port}/v1",
                    "--openai.api_key", "dummy",
                    "--env.group_size", str(args.group_size),
                    "--slurm", "false"
                ],
                cwd=atropos_dir
            )
            
            # Give environment time to register
            time.sleep(5)
            print(f"{Colors.GREEN}✓ Environment server started{Colors.ENDC}")
        
        # Build training command
        train_cmd = [
            "uv", "run", "mlx_parallm_train",
            "--model-path", args.model,
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--port", str(args.inference_port),
            "--checkpoint-dir", args.checkpoint_dir,
            "--checkpoint-interval", "1",
        ]
        
        if args.lora_path:
            train_cmd.extend(["--lora-path", args.lora_path])
        
        if not args.mock:
            train_cmd.extend([
                "--atropos-url", f"http://localhost:{args.atropos_port}",
                "--token-budget", str(args.token_budget)
            ])
        
        # Start training (includes inference server)
        mlx_dir = "/Users/shannon/Workspace/artivus/mlx_parallm"
        trainer = manager.start_service(
            "trainer",
            train_cmd,
            cwd=mlx_dir
        )
        
        # Monitor training progress
        print(f"\n{Colors.BOLD}Training in progress...{Colors.ENDC}")
        print(f"Logs are being saved to: {manager.log_dir}")
        print(f"Press Ctrl+C to stop\n")
        
        # Wait for training to complete
        trainer.wait()
        
        # Check exit code
        if trainer.returncode == 0:
            print(f"\n{Colors.GREEN}{'='*60}{Colors.ENDC}")
            print(f"{Colors.GREEN}✓ Training completed successfully!{Colors.ENDC}")
            print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")
        else:
            print(f"\n{Colors.RED}{'='*60}{Colors.ENDC}")
            print(f"{Colors.RED}✗ Training failed with exit code {trainer.returncode}{Colors.ENDC}")
            print(f"{Colors.RED}Check logs in: {manager.log_dir}{Colors.ENDC}")
            print(f"{Colors.RED}{'='*60}{Colors.ENDC}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if not args.no_cleanup:
            manager.cleanup()

if __name__ == "__main__":
    main()
