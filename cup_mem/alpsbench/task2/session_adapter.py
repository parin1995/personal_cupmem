"""Map one new_dialogue block to exactly one native CUPMEM session."""

from .schema import JsonObject, Task2Input


def build_session(example: Task2Input) -> list[JsonObject]:
    """Preserve role, text, and order across the complete incoming block.

    Splitting at turns would change CUPMEM's decision-store snapshot and its
    same-session arbitration. The runner therefore makes exactly one write
    call. Assistant turns are passed through; CUPMEM's existing user-only
    evidence selection remains responsible for selecting state evidence.
    """
    return [{"role": turn["role"], "content": turn["text"]} for turn in example.new_dialogue]
