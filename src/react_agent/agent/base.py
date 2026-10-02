"""The domain-independent ReAct loop shared by both agents.

Part 1 completes the generic loop here; the two subclasses in this package
supply only their own tools and tool executors.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
import time
import yaml
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from react_agent.env import Environment
from react_agent.agent.tools import INVOKE_SKILL_TOOL

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 1
DEFAULT_COMPACTION_MAX_TOKENS = 1_200
MAX_OBSERVATION_CHARS = 10_000

# TODO(Part 2): Write instructions that make the model produce concise working
# memory for a software agent. The prompt should preserve concrete progress,
# failures, test results, constraints, and next steps without copying raw output.
COMPACTION_SYSTEM_PROMPT = """You are a precise context compaction engine.
Generate a concise, factual working memory summary of the conversation prefix provided.
You MUST preserve:
- Objective and active constraints
- Discovered files and paths
- Executed commands and code edits
- Concrete results and observations
- Failed approaches and errors encountered
- Tests run and current test status
- Active blockers
- Next immediate action

Output ONLY the structured working memory summary without conversational filler.
"""


class StepLimitError(Exception):
    """Raised when an agent exhausts its model-call budget."""


def format_tool_output(output: dict[str, Any]) -> str:
    """Format a terminal result as a compact, tagged model observation."""

    elements: list[str] = []
    for key in sorted(output):
        value = output[key]
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_CHARS:
            # Leave room for the elision notice so the formatted value itself,
            # not just its retained source slices, stays below the limit.
            retained_at_each_end = 4_900
            omitted = len(value) - (2 * retained_at_each_end)
            value = (
                f"{value[:retained_at_each_end]}\n"
                f"[{omitted} characters elided; read a narrower range]\n"
                f"{value[-retained_at_each_end:]}"
            )
        elements.append(f"<{key}>{value}</{key}>")
    return "\n".join(elements)


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens without a provider-specific tokenizer."""

    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


class Agent:
    """Base class for a ReAct agent with pluggable tools."""

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        self.env = environment
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is not set.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        try:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        except ValueError as exc:
            raise RuntimeError("OPENAI_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise RuntimeError("OPENAI_MAX_RETRIES must be non-negative.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        if compact_threshold_tokens is not None and compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive or None")
        if (
            compaction_keep_recent_steps is not None
            and compaction_keep_recent_steps < 1
        ):
            raise ValueError("compaction_keep_recent_steps must be at least 1")
        if compaction_max_tokens is not None and compaction_max_tokens < 1:
            raise ValueError("compaction_max_tokens must be positive")
        # A None threshold turns compaction off. The other two settings then
        # describe a compaction that never happens, so fall back to the
        # defaults rather than leaving a None for later code to trip over.
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = (
            DEFAULT_COMPACTION_KEEP_RECENT_STEPS
            if compaction_keep_recent_steps is None
            else compaction_keep_recent_steps
        )
        self.compaction_max_tokens = (
            DEFAULT_COMPACTION_MAX_TOKENS
            if compaction_max_tokens is None
            else compaction_max_tokens
        )

        # Each agent supplies its own opening messages: the standing
        # instructions, and the task statement that starts the run.
        self.system_prompt: str = ""
        self.task_prompt: str = ""

        self.api_prompts: list[list[dict[str, Any]]] = []
        self.api_responses: list[dict[str, Any]] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.finished = False
        self.steps_taken = 0

        self.skills_path = Path(skills_path) if skills_path is not None else None
        self.skills: dict[str, dict[str, str]] = (
            self.load_skills(self.skills_path) if self.skills_path is not None else {}
        )

        if self.skills:
            self.tools.append(INVOKE_SKILL_TOOL)

        self.messages: list[dict[str, Any]] = [] #storage for the conversation

    def load_skills(self, skills_path: Path) -> dict[str, dict[str, str]]:
        """Load the skill folders exposed to this agent."""

        # TODO(1.4): Validate ``skills_path``, discover one ``SKILL.md``
        # per child directory, parse its YAML frontmatter (what's between the
        # `---` tags at the head of the file), and return a mapping
        # keyed by the frontmatter ``name``. Each value must contain a concise
        # ``metadata`` string for the model's skill catalog and the complete
        # ``content`` of the skill file for ``invoke_skill``. Reject duplicate
        # names and malformed or missing frontmatter with a clear
        # ``ValueError``.
        if not skills_path.exists():
            raise ValueError(f"Skills path {skills_path} does not exist.")
        if not skills_path.is_dir():
            raise ValueError(f"Skills path {skills_path} is not a directory.")

        skills: dict[str, dict[str, str]] = {}
        for skill_dir in sorted(skills_path.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                raise ValueError(f"Skill file {skill_file} does not exist.")
            if not skill_file.is_file():
                raise ValueError(f"Skill file {skill_file} is not a file.")

            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract YAML frontmatter
            if not content.startswith("---"):
                raise ValueError(f"Skill file {skill_file} is missing frontmatter.")
            try:
                _, frontmatter, _ = content.split("---", 2)
            except ValueError:
                raise ValueError(f"Skill file {skill_file} has malformed frontmatter.")

            try:
                metadata = yaml.safe_load(frontmatter)
            except yaml.YAMLError as exc:
                raise ValueError(f"Skill file {skill_file} has invalid YAML: {exc}")

            if not isinstance(metadata, dict):
                raise ValueError(f"Skill file {skill_file} frontmatter is not a dict.")

            name = metadata.get("name")
            if not name or not isinstance(name, str):
                raise ValueError(f"Skill file {skill_file} frontmatter missing 'name'.")

            if name in skills:
                raise ValueError(f"Duplicate skill name '{name}' found in {skill_file}.")

            description = metadata.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"Skill file {skill_file} frontmatter missing 'description'."
                )
            skills[name] = {
                "metadata": frontmatter.strip(),
                "content": content,
            }

        return skills

    def query_language_model(self) -> dict[str, Any]:
        """Send one tool-enabled Chat Completions request and normalize it."""

        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        step_number = self.steps_taken + 1
        print(
            f"[agent] step {step_number}/{self.step_limit}: requesting action",
            flush=True,
        )
        max_rate_limit_retries = 3
        for attempt in range(max_rate_limit_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    reasoning_effort="medium",
                    max_completion_tokens=4096,
                )
                break
            except Exception as exc:
                if (
                    attempt < max_rate_limit_retries
                    and ("RateLimitError" in type(exc).__name__ or getattr(exc, "status_code", None) == 429)
                ):
                    print(
                        f"[agent] step {step_number}: rate limit hit (429); "
                        f"backing off 12s (attempt {attempt + 1}/{max_rate_limit_retries})...",
                        flush=True,
                    )
                    time.sleep(12)
                    continue
                print(
                    f"[agent] step {step_number}: model request failed after retries "
                    f"({type(exc).__name__}: {exc})",
                    flush=True,
                )
                raise
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        message = self.process_response(response)
        tool_names = [
            call.get("function", {}).get("name", "unknown")
            for call in message.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        if tool_names:
            print(
                f"[agent] step {step_number}: tool call(s): {', '.join(tool_names)}",
                flush=True,
            )
        else:
            print(
                f"[agent] step {step_number}: response contained no parsed tool call; "
                "the loop should preserve the response and continue",
                flush=True,
            )
        return message

    def process_response(self, response: Any) -> dict[str, Any]:
        """Return relevant parts of the language model's response."""

        return response.choices[0].message.model_dump(exclude_none=True)

    def build_prompt(self) -> list[dict[str, Any]]:
        # TODO(1.1.a): Construct a sequence of messages that form the language
        # model prompt. This should include standing instructions, task
        # specification, prior interaction including observations, reasoning,
        # and actions from previous turns. Note that this method should be
        # domain-agnostic and construct the prompt in a way that would apply
        # to any of the inheriting domain-specific agents.

        # You want to be careful about which attributes of the class you modify
        # here as they may also be handled by the subclasses.
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
            *self.messages,  # include the conversation history
        ]

    def estimate_active_prompt_tokens(self) -> int:
        """Estimate the next prompt, calibrated by the provider's latest usage."""

        current_prompt = self.build_prompt()
        rough_current = rough_message_tokens(current_prompt)
        if not self.api_prompts or not self.api_responses:
            return rough_current

        usage = self.api_responses[-1].get("usage") or {}
        actual_previous = usage.get("prompt_tokens")
        if not isinstance(actual_previous, int):
            return rough_current

        rough_previous = rough_message_tokens(self.api_prompts[-1])
        added_since_previous_request = max(0, rough_current - rough_previous)
        return actual_previous + added_since_previous_request

    @property
    def compaction_enabled(self) -> bool:
        """Whether this agent compacts its context at all."""

        return self.compact_threshold_tokens is not None

    def compact_context(self):
        """Replace parts of prompt with model-generated working memory. Changes the
        content that `build_prompt` emits."""

        # TODO(2.1): Prompt the model to compact the context. The system
        # prompt should ask for concise factual working memory and preserve
        # the objective, constraints, files, commands, edits, concrete
        # results, failed approaches, tests, blockers, and next action.
        # Summarize only an old prefix; retain the original system/task
        # messages verbatim and at least the latest complete assistant action
        # with all linked tool observations. The resulting summary should change
        # what `build_prompt` emits, and reduce the length of the prompt.
        
        current_messages = self.build_prompt()

        '''
        now we first find the index of the latest assistant, and then slice self.messages based on that index.
        '''
        assistant_indices = [
            i for i, msg in enumerate(self.messages)
            if msg.get("role") == "assistant"
        ]

        # If there aren't enough assistant steps taken yet, don't compact:
        if len(assistant_indices) < self.compaction_keep_recent_steps:
            return

        
        # The cutoff is the N-th most recent assistant turn
        cutoff_index = assistant_indices[-self.compaction_keep_recent_steps]
        if cutoff_index == 0:
            return  # Nothing older than the cutoff to summarize


        header = current_messages[:2] # [system_prompt, task_prompt]
        prefix = self.messages[:cutoff_index] # older history to summarize
        suffix = self.messages[cutoff_index:] # latest assistant turn + linked tool results

        compaction_source = [*header, *prefix]


            

        compaction_prompt = [
            {
                "role": "system",
                "content": COMPACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(compaction_source),
            },
        ]

        ### Do not modify this section ###
        compaction_response = self.client.chat.completions.create(
            model=self.model,
            messages=compaction_prompt,
            reasoning_effort="medium",
            max_completion_tokens=self.compaction_max_tokens,
        )
        ##################################

        # Use `compaction_response` to update what `build_prompt` emits, but
        # DO NOT modify the object itself. Let the method return it unchanged.
        summary_text = compaction_response.choices[0].message.content or ""
        working_memory = {
            "role": "user",
            "content": f"<working_memory>\n{summary_text}\n</working_memory>",
        }
        self.messages = [working_memory, *suffix]  # replace old prefix with working memory summary
        ### Do not modify this section ###
        return compaction_prompt, compaction_response.model_dump(mode="json")
        ##################################

    def maybe_compact_context(self) -> bool:
        """Compact before the next action request when the threshold is reached."""

        if not self.compaction_enabled:
            return False

        # Context too short to compact yet
        if self.estimate_active_prompt_tokens() < self.compact_threshold_tokens:
            return False

        prompt_before = deepcopy(self.build_prompt())

        # Not enough steps (each assistant turn corresponds to a step) to force
        # compaction yet
        if (
            len([m for m in prompt_before if m.get("role") == "assistant"])
            <= self.compaction_keep_recent_steps
        ):                      
            return False

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                compaction_prompt, compaction_response = self.compact_context()
                break
            except Exception as exc:
                if attempt < max_retries and (
                    "RateLimitError" in type(exc).__name__
                    or "InternalServerError" in type(exc).__name__
                    or getattr(exc, "status_code", None) in {429, 503}
                ):
                    print(
                        f"[agent] compaction hit {type(exc).__name__} ({exc}); waiting 10s before retry ({attempt + 1}/{max_retries})...",
                        flush=True,
                    )
                    time.sleep(10)
                    continue
                raise
        prompt_after = deepcopy(self.build_prompt())
        self.compaction_events.append(
            {
                "step": self.steps_taken,
                "estimated_tokens_before": rough_message_tokens(prompt_before),
                "estimated_tokens_after": rough_message_tokens(prompt_after),
                "active_prompt_before": deepcopy(prompt_before),
                "compaction_prompt": compaction_prompt,
                "compaction_response": compaction_response,
            }
        )
        return True

    def run(self) -> None:
        """Run ReAct steps, always saving the trajectory and stopping Modal."""

        try:
            # TODO(1.2) Run the ReAct loop. Orchestrate the sequence of
            # prompting the language model to produce reasoning and actions,
            # extracting the tool calls produced by the model, and executing
            # the tool calls to obtain the agent's observation for the next
            # step. Ensure you identify when the agent has completed the task
            # by setting `Agent.finished`. If the agent exceeds the
            # `step_limit`, raise `StepLimitError`.
            while not self.finished:
                if self.steps_taken >= self.step_limit:
                    raise StepLimitError(
                        f"Agent exceeded step limit of {self.step_limit}"
                    )
                self.maybe_compact_context()

                assistant_message = self.query_language_model()
                self.messages.append(assistant_message)

                tool_calls = assistant_message.get("tool_calls", [])
                if tool_calls:
                    tool_observations = self.execute_tool_calls(tool_calls)
                    self.messages.extend(tool_observations)


        finally:
            # This block is provided infrastructure. Do not modify it: a
            # trajectory is required even when a run fails.
            if self.logs_save_path:
                path = Path(self.logs_save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "prompts": self.api_prompts,
                            "responses": self.api_responses,
                            "compactions": self.compaction_events,
                        },
                        indent=2,
                    )
                )
            if self.auto_stop_environment:
                stop = getattr(self.env, "stop", None)
                if callable(stop):
                    stop()

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute domain-specific calls and return linked tool observations."""

        # You do not need to implement anything here. This method is
        # domain-specific and implemented by the relevant subclasses
        raise NotImplementedError
