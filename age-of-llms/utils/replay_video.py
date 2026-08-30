"""Native 0 A.D. replay capture and duration fitting."""

from __future__ import annotations

import json
import math
import pathlib
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional, Union


PathLike = Union[str, pathlib.Path]


@dataclass(frozen=True)
class ReplayCapture:
    """Paths and timing produced by a native replay capture."""

    output: pathlib.Path
    raw_output: pathlib.Path
    simulation_seconds: float
    raw_seconds: float
    final_seconds: float


def replay_metadata(commands_file: PathLike) -> dict[str, Any]:
    """Read scenario metadata and the final simulation turn from commands.txt."""
    commands = pathlib.Path(commands_file)
    first_line = ""
    final_turn = 0
    turn_length_ms = 200
    with commands.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if line_number == 0:
                first_line = line.rstrip("\n")
            if line.startswith("turn "):
                parts = line.split()
                if len(parts) >= 3:
                    final_turn = max(final_turn, int(parts[1]))
                    turn_length_ms = int(parts[2])
    if not first_line.startswith("start "):
        raise ValueError(f"Unexpected replay header in {commands}")
    header = json.loads(first_line.removeprefix("start "))
    return {
        "header": header,
        "final_turn": final_turn,
        "turn_length_ms": turn_length_ms,
        "simulation_seconds": final_turn * turn_length_ms / 1000,
    }


def _duration(path: pathlib.Path) -> float:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(probe.stdout.strip())


def _stop(process: Optional[subprocess.Popen], timeout: int = 10) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def fit_video_duration(
    source: PathLike,
    destination: PathLike,
    maximum_seconds: int = 45,
    fps: int = 24,
) -> pathlib.Path:
    """Speed up a complete video when needed and retain every captured frame."""
    source_path = pathlib.Path(source)
    destination_path = pathlib.Path(destination)
    source_seconds = _duration(source_path)
    ratio = min(1.0, maximum_seconds / source_seconds)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-i",
            str(source_path),
            "-vf",
            f"setpts={ratio:.10f}*PTS,fps={fps}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination_path),
        ],
        check=True,
    )
    return destination_path


def capture_complete_replay(
    app_run: PathLike,
    commands_file: PathLike,
    destination: PathLike,
    runner: str = "zero_ad_runner",
    width: int = 960,
    height: int = 540,
    fps: int = 12,
    maximum_final_seconds: int = 45,
    playback_speed: int = 20,
    warmup_seconds: int = 55,
    renderer_safety_factor: float = 18.0,
    tail_seconds: int = 20,
    maximum_capture_seconds: int = 900,
) -> ReplayCapture:
    """Capture the full native replay, then fit every frame into a short MP4.

    Xvfb software rendering can advance more slowly than wall time. The safety
    factor allows the command stream to reach its final turn even on Colab.
    """
    app = pathlib.Path(app_run)
    commands = pathlib.Path(commands_file)
    output = pathlib.Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(f"{output.stem}.full{output.suffix}")
    metadata = replay_metadata(commands)
    simulation_seconds = float(metadata["simulation_seconds"])
    capture_seconds = min(
        maximum_capture_seconds,
        max(30, math.ceil(simulation_seconds / playback_speed * renderer_safety_factor + tail_seconds)),
    )
    if capture_seconds >= maximum_capture_seconds:
        raise RuntimeError(
            "Calculated capture duration reached the safety ceiling. Increase "
            "maximum_capture_seconds so the full replay remains covered."
        )

    display_number = next(
        number
        for number in range(99, 140)
        if not pathlib.Path(f"/tmp/.X11-unix/X{number}").exists()
    )
    display_name = f":{display_number}"
    xvfb_log = output.with_suffix(".xvfb.log").open("w")
    replay_log = output.with_suffix(".replay.log").open("w")
    xvfb = subprocess.Popen(
        [
            "Xvfb",
            display_name,
            "-screen",
            "0",
            f"{width}x{height}x24",
            "+extension",
            "GLX",
            "+render",
            "-noreset",
        ],
        stdout=xvfb_log,
        stderr=subprocess.STDOUT,
    )
    visual_engine = recorder = None
    try:
        time.sleep(2)
        visual_engine = subprocess.Popen(
            [
                "runuser",
                "-u",
                runner,
                "--",
                "env",
                f"HOME=/home/{runner}",
                f"DISPLAY={display_name}",
                "LIBGL_ALWAYS_SOFTWARE=1",
                "SDL_AUDIODRIVER=dummy",
                str(app),
                f"-replay-visual={commands}",
                f"-autostart-speed={playback_speed}",
                "-autostart-player=-1",
                f"-xres={width}",
                f"-yres={height}",
                "-conf=windowed:true",
                "-conf=pauseonfocusloss:false",
                "-conf=rendererbackend:gl",
                "-conf=vsync:false",
                "-quickstart",
                "-nosound",
            ],
            stdout=replay_log,
            stderr=subprocess.STDOUT,
        )
        for _ in range(90):
            window = subprocess.run(
                ["env", f"DISPLAY={display_name}", "xdotool", "search", "--name", "0 A.D."],
                capture_output=True,
                text=True,
            )
            if window.returncode == 0 and window.stdout.strip():
                break
            if visual_engine.poll() is not None:
                raise RuntimeError(f"Native replay exited early. Inspect {replay_log}.")
            time.sleep(1)
        else:
            raise RuntimeError("The native 0 A.D. replay window did not open.")

        time.sleep(warmup_seconds)
        subprocess.run(["env", f"DISPLAY={display_name}", "xdotool", "key", "F12"], check=False)
        recorder = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "warning",
                "-f",
                "x11grab",
                "-framerate",
                str(fps),
                "-video_size",
                f"{width}x{height}",
                "-i",
                f"{display_name}.0",
                "-t",
                str(capture_seconds),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "24",
                "-pix_fmt",
                "yuv420p",
                str(raw_output),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        first_switch = max(1, capture_seconds // 3)
        time.sleep(first_switch)
        subprocess.run(["env", f"DISPLAY={display_name}", "xdotool", "key", "Shift+Tab"], check=False)
        time.sleep(first_switch)
        subprocess.run(["env", f"DISPLAY={display_name}", "xdotool", "key", "Shift+Tab"], check=False)
        recorder.wait(timeout=capture_seconds + 45)
        if recorder.returncode != 0 or not raw_output.exists():
            raise RuntimeError("FFmpeg did not create the full native replay capture.")
    finally:
        _stop(recorder)
        _stop(visual_engine)
        _stop(xvfb)
        xvfb_log.close()
        replay_log.close()

    raw_seconds = _duration(raw_output)
    fit_video_duration(raw_output, output, maximum_final_seconds, fps=24)
    final_seconds = _duration(output)
    return ReplayCapture(output, raw_output, simulation_seconds, raw_seconds, final_seconds)
