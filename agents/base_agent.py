"""
agents/base_agent.py — Abstract BaseAgent with the Observe→Think→Act loop.

All agents in the system inherit from BaseAgent. The loop is standardised
here so every agent behaves predictably and can be monitored uniformly.

The loop terminates when:
- The agent returns a result (goal achieved)
- The max_iterations limit is hit (safety)
- A non-recoverable exception is raised

Agents can override `should_continue()` for custom termination logic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.logger import logger

# ──────────────────────────────────────────────────────────────────────────────
# Data classes used by the loop
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Observation:
    """What the agent perceives about the current state of the world."""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = ""

    def __str__(self) -> str:
        return f"Observation(source={self.source}, keys={list(self.data.keys())})"


@dataclass
class Plan:
    """What the agent intends to do next."""
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def __str__(self) -> str:
        return f"Plan(action={self.action}, reasoning={self.reasoning[:80]})"


@dataclass
class ActionResult:
    """The outcome of executing a Plan."""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    action: str = ""

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"ActionResult({status} action={self.action})"


@dataclass
class AgentResult:
    """The final result returned by an agent after its run loop completes."""
    success: bool
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract BaseAgent
# ──────────────────────────────────────────────────────────────────────────────


class BaseAgent(ABC):
    """
    Abstract base class for all Career Agent components.

    Subclasses must implement:
    - observe()  — gather current state from the world
    - think()    — decide what action to take given the observation
    - act()      — execute the decided action and return a result

    The run() method orchestrates these three methods in a loop.
    """

    # Safety limit — override in subclasses
    max_iterations: int = 50

    def __init__(self, name: str) -> None:
        self.name = name
        self._history: list[ActionResult] = []

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    async def observe(self) -> Observation:
        """Gather the current state of the world relevant to this agent."""
        ...

    @abstractmethod
    async def think(self, observation: Observation) -> Plan:
        """Decide what to do next given the current observation."""
        ...

    @abstractmethod
    async def act(self, plan: Plan) -> ActionResult:
        """Execute the plan and return the result."""
        ...

    # ── Loop control ──────────────────────────────────────────────────────────

    def should_continue(self, result: ActionResult, iteration: int) -> bool:
        """
        Determine whether the loop should continue.

        Default: continue unless action is 'done' or 'stop'.
        Override to add custom termination conditions.
        """
        return result.action not in ("done", "stop", "finished")

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self, **kwargs) -> AgentResult:
        """
        Execute the Observe→Think→Act loop.

        Returns an AgentResult summarising the entire run.
        """
        logger.info("[{}] Starting agent run", self.name)
        start_time = asyncio.get_event_loop().time()
        errors: list[str] = []
        iteration = 0

        try:
            for iteration in range(self.max_iterations):
                logger.debug("[{}] Iteration {}/{}", self.name, iteration + 1, self.max_iterations)

                # 1. Observe
                observation = await self.observe()
                logger.debug("[{}] {}", self.name, observation)

                # 2. Think
                plan = await self.think(observation)
                logger.debug("[{}] {}", self.name, plan)

                # 3. Act
                result = await self.act(plan)
                self._history.append(result)
                logger.debug("[{}] {}", self.name, result)

                if not result.success and result.error:
                    errors.append(f"[iter {iteration + 1}] {result.error}")
                    logger.warning("[{}] Action failed: {}", self.name, result.error)

                # 4. Check termination
                if not self.should_continue(result, iteration):
                    logger.info("[{}] Loop complete after {} iterations", self.name, iteration + 1)
                    break
            else:
                logger.warning("[{}] Reached max iterations ({})", self.name, self.max_iterations)
                errors.append(f"Reached max_iterations={self.max_iterations}")

        except Exception as e:
            logger.exception("[{}] Unhandled exception: {}", self.name, e)
            errors.append(str(e))
            duration = asyncio.get_event_loop().time() - start_time
            return AgentResult(
                success=False,
                summary=f"Agent {self.name} failed with exception: {e}",
                iterations=iteration,
                duration_seconds=duration,
                errors=errors,
            )

        duration = asyncio.get_event_loop().time() - start_time
        success = not errors or all(
            r.success for r in self._history[-5:]  # Last 5 actions successful
        )

        return AgentResult(
            success=success,
            summary=self._build_summary(),
            iterations=iteration + 1,
            duration_seconds=round(duration, 2),
            errors=errors,
        )

    def _build_summary(self) -> str:
        """Build a human-readable summary of the run."""
        total = len(self._history)
        succeeded = sum(1 for r in self._history if r.success)
        return f"{self.name}: {succeeded}/{total} actions succeeded"

    def history(self) -> list[ActionResult]:
        """Return the full action history for this run."""
        return list(self._history)
