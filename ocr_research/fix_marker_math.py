#!/usr/bin/env python3
"""Make Marker inline math survive a Markdown preview.

Marker writes subscripts as $SiH_4$. The preview treats "_" as italics and
drops the formula before KaTeX sees it. \\(SiH_4\\) is the same formula with
delimiters the preview does not split. Display math ($$...$$) already renders
and is left byte-for-byte. Formula text is not rewritten.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FENCE = re.compile(r"```[\s\S]*?```")
CODE = re.compile(r"`[^`\n]*`")
TOKEN = re.compile(r"\x00(\d+)\x00")


def _dollar_indexes(line: str) -> list[int]:
    indexes: list[int] = []
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "$":
            i += 2
            continue
        if line[i] == "$":
            indexes.append(i)
        i += 1
    return indexes


def _convert_inline_line(line: str) -> str:
    stripped = line.strip()
    if stripped == "$$" or (stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 2):
        return line
    indexes = _dollar_indexes(line)
    if len(indexes) < 2 or len(indexes) % 2:
        return line
    pairs = list(zip(indexes[0::2], indexes[1::2]))
    result = line
    for open_at, close_at in reversed(pairs):
        inner = result[open_at + 1 : close_at]
        result = result[:open_at] + "\\(" + inner + "\\)" + result[close_at + 1 :]
    return result


def _protect_display_blocks(text: str, saved: list[str]) -> str:
    """Keep only $$ blocks whose delimiters sit on their own lines.

    A glued pair such as ``$D_{B}$$A$`` is two inline formulas, not a block.
    Treating that ``$$`` as a block start used to skip every formula until
    the next real display equation.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "$$":
            out.append(lines[i])
            i += 1
            continue
        block = [lines[i]]
        j = i + 1
        while j < len(lines) and lines[j].strip() != "$$":
            j += 1
        if j >= len(lines):
            out.extend(lines[i:])
            break
        block.extend(lines[i + 1 : j + 1])
        i = j + 1
        chunk = "".join(block)
        newline = "\n" if chunk.endswith("\n") else ""
        saved.append(chunk[:-1] if newline else chunk)
        out.append(f"\x00{len(saved) - 1}\x00{newline}")
    return "".join(out)


def fix_text(text: str) -> str:
    saved: list[str] = []

    def hold(match: re.Match[str]) -> str:
        saved.append(match.group(0))
        return f"\x00{len(saved) - 1}\x00"

    protected = FENCE.sub(hold, text)
    protected = CODE.sub(hold, protected)
    protected = _protect_display_blocks(protected, saved)
    protected = "".join(_convert_inline_line(line) for line in protected.splitlines(keepends=True))
    return TOKEN.sub(lambda m: saved[int(m.group(1))], protected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="print counts, do not write")
    args = parser.parse_args()
    for path in args.paths:
        original = path.read_text(encoding="utf-8")
        fixed = fix_text(original)
        changed = sum(
            1
            for a, b in zip(original.splitlines(), fixed.splitlines())
            if a != b
        )
        extra = abs(original.count("\n") - fixed.count("\n"))
        print(f"{path}: lines_changed={changed + extra} inline_now={fixed.count(chr(92)+'(')}")
        if fixed == original:
            continue
        if not args.check:
            path.write_text(fixed, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
