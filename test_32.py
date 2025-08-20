import subprocess

prompts = [f"Explain concept {i} in one sentence" for i in range(1, 33)]
cmd = ["uv", "run", "python", "benchmark_concurrent.py", "--num-requests", "32", "--max-tokens", "20", "--custom-prompts"] + prompts
subprocess.run(cmd)