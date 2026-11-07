"""Review prompt-refinement proposals.

    python -m app.evolve list ["topic"]
    python -m app.evolve propose "topic"
    python -m app.evolve approve ID
    python -m app.evolve reject ID

Only approved proposals are used by later runs. No model calls are made here.
"""

import sys

from app.memory.database import MemoryDatabaseError
from app.memory.evolution import approve, propose_refinements, reject
from app.memory.repository import MemoryRepository, PromptVersion

USAGE = __doc__ or ""


def _show(version: PromptVersion) -> None:
    print(
        f"#{version.id} [{version.status}] {version.topic} v{version.version}: "
        f"{version.failure_pattern}"
    )
    print(f"    why:      {version.rationale}")
    print(f"    previous: {version.previous_instruction or '(none)'}")
    print(f"    proposed: {version.proposed_instruction}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in {"list", "propose", "approve", "reject"}:
        print(USAGE, file=sys.stderr)
        return 2
    command, rest = args[0], args[1:]
    try:
        memory = MemoryRepository.from_settings()
        if command == "list":
            versions = memory.list_prompt_versions(" ".join(rest) or None)
            for version in versions:
                _show(version)
            if not versions:
                print("No prompt versions stored.")
        elif command == "propose":
            if not rest:
                print(USAGE, file=sys.stderr)
                return 2
            created = propose_refinements(memory, " ".join(rest))
            for version in created:
                _show(version)
            if not created:
                print("No new repeated failure patterns.")
        else:
            if len(rest) != 1 or not rest[0].isdigit():
                print(USAGE, file=sys.stderr)
                return 2
            action = approve if command == "approve" else reject
            _show(action(memory, int(rest[0])))
    except (MemoryDatabaseError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
