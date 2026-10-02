"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    for attempt in range(3):
        try:
            response = client.request(method, endpoint, **kwargs)
            break
        except (httpx.TransportError, httpx.RemoteProtocolError):
            if attempt == 2:
                raise
            import time
            time.sleep(1)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    # TODO(Part 3.3.b): Parse the arguments, call the provided
    # /api/simulate endpoint with fen and optional move, and return its JSON.
    # Catch any errors raised by the tool and return an error message between
    # `<chess_error></chess_error>` for the agent to address. Cover malformed
    # JSON arguments, arguments that are not an object, a missing or
    # non-string fen, a non-string move, a position or move the server rejects,
    # and a transport failure.
    try:
        args = json.loads(arguments)
    except Exception as e:
        return f"<chess_error>Error parsing arguments: {str(e)}</chess_error>"

    if not isinstance(args, dict):
        return "<chess_error>Arguments must be a JSON object.</chess_error>"
    
    if "fen" not in args:
        return "<chess_error>Missing required argument: fen</chess_error>"

    fen = args["fen"]
    if not isinstance(fen, str):
        return "<chess_error>Fen must be a string</chess_error>"
    move = args.get("move", None)
    if not (move is None or isinstance(move, str)):
        return "<chess_error>move must be a string or null</chess_error>"
    
    try:
        payload = {"fen": fen}
        if move is not None:
            payload["move"] = move
        response = client.post("/api/simulate", json=payload)
        response.raise_for_status()
        return json.dumps(response.json())
    except httpx.HTTPStatusError as e:
        return f"<chess_error>Illegal move: {e.response.text}</chess_error>"
    except Exception as e:
        return f"<chess_error>Error playing move: {str(e)}</chess_error>"
    


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    # TODO(3.1.b): Parse the arguments and POST {"move": <uci move>} to
    # /api/move. Return its JSON object. Catch any errors raised by the
    # tool and return an error message between `<chess_error></chess_error>`
    # for the agent to address. Cover malformed JSON arguments, arguments
    # that are not an object, a missing or non-string fen, a non-string move,
    # a position or move the server rejects, and a transport failure.
    try:
        args = json.loads(arguments)
    except Exception as e:
        return f"<chess_error>Error parsing arguments: {str(e)}</chess_error>"

    if not isinstance(args, dict):
        return "<chess_error>Arguments must be a JSON object.</chess_error>"
    
    if "move" not in args:
        return "<chess_error>Missing required argument: move</chess_error>"

    move = args["move"]
    if not isinstance(move, str):
        return "<chess_error>Move must be a string</chess_error>"

    try:
        response = client.post("/api/move", json={"move": move})
        response.raise_for_status()
        return json.dumps(response.json())
    except httpx.HTTPStatusError as e:
        return f"<chess_error>Illegal move: {e.response.text}</chess_error>"
    except Exception as e:
        return f"<chess_error>Error playing move: {str(e)}</chess_error>"
    


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    # TODO(3.4): parse the arguments and run the code in the
    # sandbox with the registered tools available by name.
    #
    # `/opt/react_agent/sandbox_python.py` is a script on the `env` sandbox
    # that has access to the same tool definitions in this file. Use it to run
    # the code that the model produced as an argument to the run_python tool.
    # The script accepts two positional arguments -- `port` and a base64-encoded
    # string of code (to prevent issues with quoting). Implement this tool
    # call.
    #
    # The script prints one JSON object with `stdout`, `stderr`, and `error`
    # from running the code -- return that string as it is.
    #
    # A non-zero returncode means the sandbox itself failed, not the model's
    # code. Report `exception_info` or `stderr` as a <chess_error>.
    #
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    try:
        args = json.loads(arguments)
    except Exception as e:
        return f"<chess_error>Error parsing arguments: {str(e)}</chess_error>"

    if not isinstance(args, dict):
        return "<chess_error>Arguments must be a JSON object.</chess_error>"
    
    if "code" not in args:
        return "<chess_error>Missing required argument: code</chess_error>"
    
    code = args["code"]
    if not isinstance(code, str):
        return "<chess_error>Code must be a string</chess_error>"
    
    try:
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        command = f"python /opt/react_agent/sandbox_python.py {port} {encoded_code}"
        result = env.execute(command)

        if result.get("returncode") != 0:
            err = result.get("exception_info") or result.get("output", "Execution failed")
            return f"<chess_error>{err}</chess_error>"

        return result.get("output", "")
    except Exception as e:
        return f"<chess_error>Error running code: {str(e)}</chess_error>"

    
    


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    # TODO(3.5): parse the arguments and return the named skill's content.
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    try:
        args = json.loads(arguments)
    except Exception as e:
        return f"<chess_error>Error parsing arguments: {str(e)}</chess_error>"

    if not isinstance(args, dict):
        return "<chess_error>Arguments must be a JSON object.</chess_error>"
    
    if "name" not in args:
        return "<chess_error>Missing required argument: name</chess_error>"
    
    skill_name = args["name"]
    if not isinstance(skill_name, str):
        return "<chess_error>Skill name must be a string</chess_error>"
    
    skill = skills.get(skill_name)
    if not skill:
        return f"<chess_error>Skill '{skill_name}' not found.</chess_error>"
    
    return skill.get("content", "")


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
