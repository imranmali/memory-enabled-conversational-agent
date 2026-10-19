"""``memory-agent``: chat with the memory-enabled agent from a terminal or a pipe.

Each input line is one user message; replies are printed prefixed with ``assistant> ``.
Lines starting with ``/`` are commands (see ``/help``). On ``/quit`` or end of input the
session is summarized, synchronized and closed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any, TextIO

from .agent import AgentConfig, MemoryAgent, default_device_id
from .llm import LLM_NAMES, build_llm

__all__ = ["main"]

PREFIX = "assistant> "
PROMPT = "you> "
DEFAULT_DATA_DIR = "~/.memory-agent"

HELP_TEXT = """Commands:
  /help              show this help
  /facts             list what I know about you
  /memories [query]  list memories (or recall the ones relevant to a query)
  /forget-all        forget everything about you
  /sync              synchronize memories with other devices
  /stats             show memory and session statistics
  /quit              end the session and exit (also /exit or end of input)"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory-agent", description="Chat with an assistant that remembers you."
    )
    parser.add_argument("--user", required=True, help="user id whose memories to use")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"directory for memories and sync data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--llm", default="offline", choices=LLM_NAMES, help="language model (default: offline)"
    )
    parser.add_argument("--model", default=None, help="model name for the anthropic provider")
    parser.add_argument(
        "--device",
        default=None,
        help="device id used for sync (default: derived from the host name)",
    )
    return parser


def _say(out: TextIO, text: str) -> None:
    for i, line in enumerate(text.splitlines() or [""]):
        print(f"{PREFIX if i == 0 else ' ' * len(PREFIX)}{line}", file=out)
    out.flush()


def _command(agent: MemoryAgent, line: str, out: TextIO) -> bool:
    """Run a slash command; return False when the session should end."""
    name, _, arg = line.partition(" ")
    name = name.lower()
    arg = arg.strip()
    if name in ("/quit", "/exit"):
        return False
    if name == "/help":
        _say(out, HELP_TEXT)
    elif name == "/facts":
        facts = agent.known_facts()
        _say(out, "\n".join(f"- {f}" for f in facts) if facts else "I don't know any facts yet.")
    elif name == "/memories":
        _memories(agent, arg, out)
    elif name == "/forget-all":
        count = agent.forget_everything()
        _say(out, f"Forgot {count} memories.")
    elif name == "/sync":
        report = agent.sync()
        if report is None:
            _say(out, "Sync is not configured.")
        else:
            _say(
                out,
                f"Synced: pushed {report.pushed}, pulled {report.pulled}, "
                f"conflicts {report.conflicts}, revision {report.revision}.",
            )
    elif name == "/stats":
        _stats(agent, out)
    else:
        _say(out, f"Unknown command: {name}. Type /help for the list of commands.")
    return True


def _memories(agent: MemoryAgent, query: str, out: TextIO) -> None:
    if query:
        recalled = agent.memory.recall(agent.user_id, query, k=10, diversify=False)
        lines = [f"- [{s.score:.2f}] ({s.item.kind}) {s.item.content}" for s in recalled]
    else:
        items = agent.memory.memories(agent.user_id)[-20:]
        lines = [f"- ({i.kind}) {i.content}" for i in items]
    _say(out, "\n".join(lines) if lines else "No memories found.")


def _stats(agent: MemoryAgent, out: TextIO) -> None:
    session = agent.session
    lines = [
        f"user: {agent.user_id}",
        f"device: {agent.config.device_id}",
        f"memories: {agent.memory.count(agent.user_id)}",
        f"facts: {len(agent.known_facts())}",
        f"short-term messages: {len(agent.buffer)} ({agent.buffer.token_count()} tokens)",
        f"session: {session.session_id if session else 'none'}",
    ]
    if session is not None and agent.session_manager is not None:
        lines.append(
            f"session messages: {agent.session_manager.get(session.session_id).message_count}"
        )
    if agent.data_dir is not None:
        lines.append(f"data dir: {agent.data_dir}")
    _say(out, "\n".join(lines))


def main(
    argv: Sequence[str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run the chat loop; returns the process exit status."""
    args = build_parser().parse_args(argv)
    inp = stdin if stdin is not None else sys.stdin
    out = stdout if stdout is not None else sys.stdout

    llm_kwargs: dict[str, Any] = {}
    if args.llm == "anthropic" and args.model:
        llm_kwargs["model"] = args.model
    try:
        llm = build_llm(args.llm, **llm_kwargs)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    device = args.device or default_device_id()
    try:
        config = AgentConfig(device_id=device)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    interactive = bool(getattr(inp, "isatty", lambda: False)())
    agent = MemoryAgent.from_data_dir(args.user, args.data_dir, llm=llm, config=config)
    try:
        agent.start_session()
        _say(out, agent.greeting())
        while True:
            if interactive:
                print(PROMPT, end="", file=out, flush=True)
            raw = inp.readline()
            if not raw:
                break
            line = raw.strip()
            if not line:
                continue
            if line.startswith("/"):
                if not _command(agent, line, out):
                    break
                continue
            _say(out, agent.chat(line).text)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print(file=out)
    finally:
        agent.close()
    _say(out, "Goodbye! I'll remember our conversation.")
    return 0
