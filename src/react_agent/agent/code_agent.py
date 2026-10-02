"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
import textwrap
import shlex
import time
from typing import Any

from react_agent.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from react_agent.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from react_agent.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
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
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])


        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        system_information = json.dumps(
            {
                "machine": environment.machine,
                "release": environment.release,
                "system": environment.system,
                "version": environment.version,
            },
            indent=2,
        )

        self.system_prompt = textwrap.dedent(f"""\
        You are a coding agent. Solve the user's software task by inspecting the repository, editing files, running tests, and producing the required patch.

        <system_information>
        {system_information}
        </system_information>

        Use the execute tool to work in the sandbox. When the task is complete, use send_message to report completion.
        """)


        self.task_prompt = self.task

        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        if self.skills:
            catalog = "\n".join(
                skill["metadata"] for skill in self.skills.values()
            )
            self.system_prompt += (
                "\nAvailable skills:\n"
                "<skills>\n"
                f"{catalog}\n"
                "</skills>\n"
                "Use invoke_skill to load a skill's full instructions."
            )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations = []

        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name")
            raw_arguments = function.get("arguments", "")

            try:
                if name == "execute":
                    arguments = json.loads(raw_arguments)

                    if not isinstance(arguments, dict):
                        raise ValueError("Arguments must be a JSON object.")
                    # fix: If the model passes command as a list, join it properly for the shell
                    command = arguments.get("command")
                    if isinstance(command, list):
                        arguments["command"] = shlex.join(command)
                    # fix: Default safety timeout (e.g., 60s) so bad commands don't hang for 10 minutes

                    if not arguments.get("timeout"):
                        arguments["timeout"] = 60

                    result = self.env.execute(**arguments)
                    content = format_tool_output(result)
                elif name == "invoke_skill":
                    arguments = json.loads(raw_arguments)

                    if not isinstance(arguments, dict):
                        raise ValueError("Arguments must be a JSON object.")

                    skill_name = arguments.get("name")
                    if not isinstance(skill_name, str):
                        raise ValueError("invoke_skill requires a string name.")

                    skill = self.skills.get(skill_name)
                    if skill is None:
                        raise ValueError(f"Unknown skill: {skill_name}")

                    content = skill["content"]
                elif name == "send_message":
                    arguments = json.loads(raw_arguments)

                    if not isinstance(arguments, dict):
                        raise ValueError("Arguments must be a JSON object.")

                    summary = arguments.get("summary")
                    if not isinstance(summary, str):
                        raise ValueError("send_message requires a string summary.")

                    # Handle the completed message here.
                    self.finished = True
                    content = summary

                else:
                    content = f"<tool_error>Unknown tool: {name}</tool_error>"

            except Exception as exc:
                content = f"<tool_error>{exc}</tool_error>"

            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": content,
                }
            )

        return observations


