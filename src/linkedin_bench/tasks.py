"""
Task definitions for LinkedIn browser automation benchmark.

Each task has:
- id: Unique identifier
- prompt: What the agent should do
- expected: Natural language description of expected result (for LLM judge)
- timeout: Maximum seconds allowed for the task
"""

from dataclasses import dataclass


@dataclass
class Task:
    """A browser automation task to evaluate."""

    id: str
    prompt: str
    expected: str
    timeout: int = 120


# The three LinkedIn tasks for this benchmark
TASKS = [
    Task(
        id="own_follower_count",
        prompt="Use your linkedin skill with agent-browser to navigate to my LinkedIn profile and get my follower count. Return just the number.",
        expected="Around 2001 followers (accept within ±10%, so 1800-2200 is correct)",
        timeout=300,  # Browser automation can be slow, allow 5 minutes
    ),
    Task(
        id="own_connection_count",
        prompt="Use your linkedin skill with agent-browser to navigate to my LinkedIn profile and get my connection count. Return just the number.",
        expected="Around 653 connections (accept within ±5%, so 620-686 is correct)",
        timeout=300,
    ),
    Task(
        id="search_bill_gates_followers",
        prompt="Use your linkedin skill with agent-browser to search for 'Bill Gates' on LinkedIn, navigate to his profile, and get his follower count. Return just the number.",
        expected="Around 39.8 million followers (accept within ±5%, so roughly 37.8M-41.8M is correct)",
        timeout=360,  # Search + navigate takes longer
    ),
]

# Map task IDs to tasks for easy lookup
TASK_BY_ID = {task.id: task for task in TASKS}


def get_task_by_seed(seed: int) -> Task:
    """Get a task by seed number (wraps around)."""
    return TASKS[seed % len(TASKS)]


def get_task_by_id(task_id: str) -> Task:
    """Get a task by its ID."""
    if task_id not in TASK_BY_ID:
        raise ValueError(f"Unknown task ID: {task_id}. Available: {list(TASK_BY_ID.keys())}")
    return TASK_BY_ID[task_id]
