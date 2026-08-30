"""Artifact checksums and replay packaging."""

from __future__ import annotations

import hashlib
import pathlib
import shutil
from typing import Union


PathLike = Union[str, pathlib.Path]


def file_sha256(path: PathLike) -> str:
    """Return the SHA-256 checksum for a file."""
    source = pathlib.Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_replay_archive(
    replay_directory: PathLike,
    destination_without_suffix: PathLike,
) -> pathlib.Path:
    """Create a ZIP that preserves the native 0 A.D. replay directory."""
    replay = pathlib.Path(replay_directory).resolve()
    destination = pathlib.Path(destination_without_suffix).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(
        str(destination),
        "zip",
        root_dir=replay.parent,
        base_dir=replay.name,
    )
    return pathlib.Path(archive)
