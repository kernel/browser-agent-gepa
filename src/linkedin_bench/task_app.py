"""
FastAPI Task App for LinkedIn Browser Automation Benchmark.

This task app is called by the Synth backend during GEPA optimization.
Uses the synth_ai SDK's LocalAPI pattern for proper integration.
"""

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import Request
from kernel import AsyncKernel

# Import Synth SDK types for proper protocol compatibility
from synth_ai.data.artifacts import Artifact
from synth_ai.data.enums import SuccessStatus
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi._impl.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)

from .kernel_runner import (
    DEFAULT_POOL_NAME,
    DEFAULT_POOL_SIZE,
    ensure_browser_ready,
    get_or_create_pool,
    run_claude_code,
    write_skill_file,
)
from .skill_template import get_skill_content
from .tasks import TASKS, Task, get_task_by_seed
from .verifier import (
    VerificationResult,
    calculate_reward,
    count_agent_steps,
    count_sleep_commands,
    verify_with_llm,
)

# Load environment variables
load_dotenv()

# App configuration
APP_ID = "linkedin_bench"
APP_NAME = "LinkedIn Browser Automation Benchmark"

# Global Kernel client
kernel_client: AsyncKernel | None = None


def provide_taskset_description() -> str:
    """Provide description of the task set."""
    return """LinkedIn Browser Automation Benchmark

This benchmark evaluates browser automation skills for LinkedIn tasks:
- Getting follower counts from profiles
- Getting connection counts
- Searching for and navigating to user profiles

The agent uses Claude Code with agent-browser CLI to automate Chrome.
"""


def provide_task_instances() -> list[TaskInfo]:
    """Provide list of task instances."""
    return [
        TaskInfo(
            id=str(i),
            name=task.id,
            description=task.prompt,
        )
        for i, task in enumerate(TASKS)
    ]


async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    """
    Handle a single evaluation rollout.

    This runs the full pipeline:
    1. Acquire browser from Kernel pool
    2. Write skill file (which GEPA optimizes)
    3. Run Claude Code with task prompt
    4. Verify result with LLM judge
    5. Return outcome reward
    """
    global kernel_client

    seed = request.env.seed or 0
    policy_config = request.policy.config if request.policy else {}

    # Get the task for this seed
    task = get_task_by_seed(seed)

    # Use task-specific timeout
    timeout = task.timeout or policy_config.get("timeout", 120)

    # Get skill content from context_overrides (GEPA provides optimized versions)
    context = request.context_overrides or {}
    skill_content = context.get("skill_content", get_skill_content())

    print(f"\n{'=' * 60}")
    print(f"Rollout: seed={seed}, task={task.id}")
    print(f"Timeout: {timeout}s")
    print(f"{'=' * 60}")

    # Initialize Kernel client if needed
    if kernel_client is None:
        api_key = os.environ.get("KERNEL_API_KEY")
        if not api_key:
            return RolloutResponse(
                reward_info=RolloutMetrics(
                    outcome_reward=0.0,
                    details={"error": "KERNEL_API_KEY not set"},
                ),
                success_status=SuccessStatus.FAILURE,
                status_detail="KERNEL_API_KEY not set",
            )
        kernel_client = AsyncKernel(api_key=api_key)

        # Ensure browser pool exists
        try:
            await get_or_create_pool(kernel_client, DEFAULT_POOL_NAME, "linkedin", DEFAULT_POOL_SIZE)
        except Exception as e:
            print(f"Warning: Could not create browser pool: {e}")

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        return RolloutResponse(
            reward_info=RolloutMetrics(
                outcome_reward=0.0,
                details={"error": "ANTHROPIC_API_KEY not set"},
            ),
            success_status=SuccessStatus.FAILURE,
            status_detail="ANTHROPIC_API_KEY not set",
        )

    session_id = None
    start_time = time.time()

    try:
        # Acquire browser from pool
        print(f"Acquiring browser from pool '{DEFAULT_POOL_NAME}'...")
        acquire_result = await kernel_client.browser_pools.acquire(DEFAULT_POOL_NAME)
        session_id = acquire_result.session_id
        print(f"  Session: {session_id}")
        print(f"  Live view: {acquire_result.browser_live_view_url}")

        # Ensure browser is ready
        print("Ensuring browser is ready...")
        await ensure_browser_ready(kernel_client, session_id)

        # Write skill file
        print("Writing skill file...")
        await write_skill_file(kernel_client, session_id, skill_content)

        # Run Claude Code
        print(f"Running Claude Code...")
        run_result = await run_claude_code(
            kernel_client,
            session_id,
            task.prompt,
            anthropic_api_key,
            timeout=timeout,
        )
        elapsed = time.time() - start_time

        print(f"Claude finished in {elapsed:.1f}s (exit code: {run_result.exit_code})")
        print(f"Output length: {len(run_result.output)} chars")

        # Count agent steps and sleep commands
        num_steps = count_agent_steps(run_result.output)
        num_sleeps, total_sleep_ms = count_sleep_commands(run_result.output)
        print(f"Agent steps: {num_steps}")
        if num_sleeps > 0:
            print(f"Sleep commands: {num_sleeps} ({total_sleep_ms}ms total) - PENALIZED")

        # Verify result
        print("Verifying result with LLM judge...")
        verification = await verify_with_llm(task, run_result.output, anthropic_api_key)
        print(f"  Correct: {verification.correct}")
        print(f"  Extracted: {verification.extracted_answer}")
        print(f"  Reason: {verification.reason}")

        # Calculate reward (with sleep penalty)
        reward = calculate_reward(
            correctness_score=verification.raw_score,
            elapsed_seconds=elapsed,
            num_agent_steps=num_steps,
            num_sleep_commands=num_sleeps,
            total_sleep_ms=total_sleep_ms,
            max_time=float(timeout),
            max_steps=20,
        )
        print(f"Final reward: {reward:.3f}")

        return RolloutResponse(
            reward_info=RolloutMetrics(
                outcome_reward=reward,
                details={
                    "task_id": task.id,
                    "correct": verification.correct,
                    "extracted_answer": verification.extracted_answer,
                    "verification_reason": verification.reason,
                    "elapsed_seconds": elapsed,
                    "num_steps": num_steps,
                    "num_sleeps": num_sleeps,
                    "total_sleep_ms": total_sleep_ms,
                    "exit_code": run_result.exit_code,
                },
                outcome_objectives={"reward": reward, "latency_ms": int(elapsed * 1000)},
            ),
            success_status=SuccessStatus.SUCCESS,
            status_detail=f"Correct: {verification.correct}, Reward: {reward:.3f}",
            artifact=[
                Artifact(
                    content=run_result.output[-10000:],  # Last 10k chars
                    content_type="agent_output",
                    metadata={"task_id": task.id},
                )
            ],
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        elapsed = time.time() - start_time
        print(f"Rollout failed: {e}")

        return RolloutResponse(
            reward_info=RolloutMetrics(
                outcome_reward=0.0,
                details={"error": str(e), "task_id": task.id, "elapsed_seconds": elapsed},
            ),
            success_status=SuccessStatus.FAILURE,
            status_detail=f"Error: {str(e)[:200]}",
        )

    finally:
        # Release browser back to pool
        if session_id and kernel_client:
            print(f"Releasing browser {session_id}...")
            try:
                await kernel_client.browser_pools.release(
                    DEFAULT_POOL_NAME, session_id=session_id, reuse=True
                )
                print("  Released with reuse=True")
            except Exception as e:
                print(f"  Warning: Failed to release: {e}")


# Create the app using Synth's LocalAPI helper
app = create_local_api(
    LocalAPIConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description="Evaluate browser automation skills on LinkedIn tasks using Claude Code and agent-browser.",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        cors_origins=["*"],
    )
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8030"))
    uvicorn.run(app, host="0.0.0.0", port=port)
