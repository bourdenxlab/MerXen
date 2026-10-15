"""Sample names in the acceptance scripts' inputs and outputs (privacy rules).

The scripts key samples by their aliases (``ms1``, ``ms2``, ``HD1`` ...;
docs/local-configuration.md). At their edges:

- ``alias_key`` (rule R2) translates a key read from a command line or a
  recorded file (``--dataset <name>=ROOT``, an exception's scope, the gene-set
  keys of a recorded ``mo10.json``), real name or alias, to the script's key;
- ``display_tree`` / ``display_text`` (rule R5) write labels as
  ``privacy.display`` shows them: on a host with the alias table the names
  the outputs carried before 2026-10-14, so a re-run is byte-identical; with
  the table disabled (tests, CI) the aliases. Absolute paths inside output
  text are kept verbatim (a path is data, never a label).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

from merxen import privacy

# An absolute path inside a text: from a '/' that does not follow a word
# character, '.' or '~', up to whitespace, a quote, a bracket or a separator.
_ABSOLUTE_PATH: Final = re.compile(r"(?<![\w.~])/[^\s\"'()\[\]{},;]*")


def alias_key(text: str, keys: Iterable[str] = ()) -> str:
    """Return the script's key for a key read from outside the repository.

    Args:
        text: A real name or an alias (``privacy.to_alias`` translates a real
            name; an alias passes unchanged).
        keys: The script's keys: a result that equals one of them ignoring
            case is returned in the key's own spelling (``to_alias`` keeps
            the case pattern of the text it translates).

    Returns:
        The key.

    Raises:
        LocalConfigError: If no alias table is found and it is not disabled.
    """
    aliased = privacy.to_alias(text)
    for key in keys:
        if aliased.casefold() == key.casefold():
            return key
    return aliased


def display_text(text: str) -> str:
    """Return ``privacy.display(text)`` with absolute paths left verbatim."""
    if not privacy.has_aliases():
        return privacy.display(text)
    pieces: list[str] = []
    cursor = 0
    for match in _ABSOLUTE_PATH.finditer(text):
        pieces.append(privacy.display(text[cursor : match.start()]))
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(privacy.display(text[cursor:]))
    return "".join(pieces)


def display_tree(value: Any) -> Any:
    """Return a JSON-like value with every string (and key) through ``display_text``.

    Args:
        value: Mappings, lists, tuples, strings and scalars.

    Returns:
        A copy (mappings become dicts, tuples lists); other values unchanged.
    """
    if isinstance(value, str):
        return display_text(value)
    if isinstance(value, Mapping):
        return {
            display_text(key) if isinstance(key, str) else key: display_tree(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [display_tree(item) for item in value]
    return value
