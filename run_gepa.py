#!/usr/bin/env python3
"""
Run GEPA optimization on LinkedIn browser automation skill.

This script:
1. Starts the task app locally
2. Creates a SynthTunnel to expose it to Synth's infrastructure
3. Submits a GEPA job to the Synth backend
4. Polls for results
5. Saves the optimized skill

Usage:
    # Run with SynthTunnel (default - currently broken, see problems/synthtunnel.md)
    uv run python run_gepa.py

    # Workaround: Use ngrok instead
    ngrok http 8030 --url your-subdomain.ngrok-free.app
    TASK_APP_URL=https://your-subdomain.ngrok-free.app uv run python run_gepa.py

    # Override generations and budget
    uv run python run_gepa.py --generations 5 --budget 50

    # Use local Synth backend (no tunnel needed)
    uv run python run_gepa.py --local
"""

import argparse
import asyncio
import json
import os
import sys
import time
import tomllib
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from linkedin_bench.skill_template import get_skill_content

load_dotenv()


def wait_for_health(host: str, port: int, api_key: str = "", timeout: float = 30.0) -> None:
    """Wait for task app to be healthy."""
    import httpx

    url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, headers=headers, timeout=5.0)
            if r.status_code in (200, 400):
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Health check failed: {url}")


async def main():
    parser = argparse.ArgumentParser(description="Run GEPA optimization for LinkedIn skill")
    parser.add_argument("--local", action="store_true", help="Use localhost:8000 Synth backend")
    parser.add_argument("--port", type=int, default=8030, help="Task app port (default: 8030)")
    parser.add_argument("--config", type=str, default="linkedin_gepa.toml", help="GEPA config file")
    parser.add_argument("--generations", type=int, help="Override number of generations")
    parser.add_argument("--budget", type=int, help="Override rollout budget")
    parser.add_argument("--timeout", type=int, default=120, help="Agent timeout per rollout")
    args = parser.parse_args()

    # Check environment
    required_vars = ["SYNTH_API_KEY", "KERNEL_API_KEY", "ANTHROPIC_API_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    synth_api_key = os.environ["SYNTH_API_KEY"]
    backend_url = "http://localhost:8000" if args.local else "https://api.usesynth.ai"

    print("=" * 60)
    print("GEPA OPTIMIZATION - LinkedIn Browser Automation Skill")
    print("=" * 60)
    print(f"Backend: {backend_url}")

    # Get environment API key from Synth backend
    print("Authenticating with Synth backend...")
    try:
        from synth_ai.sdk.localapi.auth import ensure_localapi_auth
        environment_api_key = ensure_localapi_auth(
            backend_base=backend_url,
            synth_api_key=synth_api_key,
        )
        print(f"Environment key ready: {environment_api_key[:15]}...")
    except Exception as e:
        print(f"Warning: Could not get environment API key: {e}")
        environment_api_key = synth_api_key
    print(f"Config: {args.config}")

    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Apply overrides
    if args.generations:
        config["prompt_learning"]["gepa"]["population"]["num_generations"] = args.generations
    if args.budget:
        config["prompt_learning"]["gepa"]["rollout"]["budget"] = args.budget


    # Inject the initial skill content into the wildcard
    # GEPA needs to know the initial value to mutate from
    initial_skill = get_skill_content()
    config["prompt_learning"]["initial_prompt"]["wildcards"]["skill_content"] = initial_skill
    print(f"Initial skill content: {len(initial_skill)} chars")

    print(f"Generations: {config['prompt_learning']['gepa']['population']['num_generations']}")
    print(f"Budget: {config['prompt_learning']['gepa']['rollout']['budget']}")
    print(f"Timeout: {args.timeout}s")

    # Start task app in background
    print("\nStarting task app...")
    import threading
    import uvicorn

    # Import the app from our module
    from linkedin_bench.task_app import app

    port = args.port

    def run_server():
        config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        server.run()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for task app to be ready
    wait_for_health("localhost", port, environment_api_key)
    print(f"Task app ready on port {port}")

    # Set up tunnel or use override URL
    tunnel = None
    worker_token = None
    task_app_url_override = os.environ.get("TASK_APP_URL")

    if task_app_url_override:
        # Manual override (e.g., ngrok)
        task_app_url = task_app_url_override
        print(f"Task app URL (from env): {task_app_url}")
    elif args.local:
        # Local backend can reach localhost directly
        task_app_url = f"http://localhost:{port}"
        print(f"Task app URL (local): {task_app_url}")
    else:
        # Use SynthTunnel to expose local task app to remote backend
        print("\nCreating SynthTunnel...")
        from synth_ai.core.tunnels import TunneledLocalAPI

        tunnel = await TunneledLocalAPI.create(
            local_port=port,
            api_key=synth_api_key,
        )
        task_app_url = tunnel.url
        worker_token = tunnel.worker_token
        print(f"Tunnel URL: {task_app_url}")
        print(f"Worker token: {worker_token[:20]}...")

    config["prompt_learning"]["task_app_url"] = task_app_url

    try:
        # Submit GEPA job
        print("\nSubmitting GEPA job...")

        # Import Synth SDK
        try:
            from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob
        except ImportError:
            print("ERROR: synth-ai SDK not installed. Install with: pip install synth-ai")
            sys.exit(1)

        # Use worker_token for SynthTunnel, environment_api_key otherwise
        job_kwargs = {
            "config_dict": config,
            "backend_url": backend_url,
            "api_key": synth_api_key,
            "skip_health_check": True,
        }
        if worker_token:
            job_kwargs["task_app_worker_token"] = worker_token
        else:
            job_kwargs["task_app_api_key"] = environment_api_key

        job = PromptLearningJob.from_dict(**job_kwargs)

        job_id = job.submit()
        print(f"Job submitted: {job_id}")

        # Poll for results with progress updates
        print("\nPolling for results (this may take a while)...")
        result = job.poll_until_complete(
            timeout=7200.0,  # 2 hours max
            interval=15.0,
            progress=True,
        )

        print("\n" + "=" * 60)
        print("GEPA RESULTS")
        print("=" * 60)
        print(f"Status: {result.status}")

        if result.failed:
            print(f"Job failed: {result.error}")
            if result.raw:
                print(f"Raw response: {json.dumps(result.raw, indent=2)[:2000]}")
        else:
            if result.best_score is not None:
                print(f"Best score: {result.best_score:.4f}")

            # Save the optimized skill
            if result.best_prompt:
                output_dir = Path(__file__).parent / "output"
                output_dir.mkdir(exist_ok=True)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"optimized_skill_{timestamp}.md"

                if isinstance(result.best_prompt, dict):
                    # Extract skill content from messages
                    messages = result.best_prompt.get("messages", [])
                    for msg in messages:
                        if msg.get("role") == "system":
                            skill_content = msg.get("pattern", "")
                            output_file.write_text(skill_content)
                            print(f"\nOptimized skill saved to: {output_file}")
                            print("\nFirst 500 chars of optimized skill:")
                            print("-" * 40)
                            print(skill_content[:500])
                            break
                elif isinstance(result.best_prompt, str):
                    output_file.write_text(result.best_prompt)
                    print(f"\nOptimized skill saved to: {output_file}")

            # Print training stats
            if result.raw:
                best_train = result.raw.get("best_train_score")
                best_val = result.raw.get("best_validation_score")
                if best_train is not None:
                    print(f"\nBest train score: {best_train:.4f}")
                if best_val is not None:
                    print(f"Best validation score: {best_val:.4f}")

    finally:
        # Cleanup tunnel if created
        if tunnel:
            print("\nClosing SynthTunnel...")
            tunnel.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
