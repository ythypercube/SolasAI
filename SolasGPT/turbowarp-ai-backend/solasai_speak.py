#!/usr/bin/env python3
"""
SolasAI TTS helper — called by the Fabric mod to speak AI replies.
Uses edge-tts (Microsoft neural voices, requires internet).
Falls back to espeak-ng if offline.

Usage:
    python3 solasai_speak.py "Hello, I am SolasAI!"
    python3 solasai_speak.py --voice en-US-GuyNeural "Hello!"
"""

import sys
import os
import subprocess
import tempfile
import asyncio
import argparse

# Nice, energetic voice — sounds like a gaming AI/assistant
DEFAULT_VOICE = os.environ.get("SOLASAI_VOICE", "en-US-GuyNeural")


async def speak_edge(text: str, voice: str) -> bool:
    try:
        import edge_tts  # type: ignore
        tts = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        await tts.save(tmp_path)
        # Try players in order. mpg123 handles MP3 directly.
        for player in (
            "mpg123 -q",
            "ffplay -nodisp -autoexit -loglevel quiet",
            "mpv --no-terminal",
            "mplayer -really-quiet",
        ):
            cmd = player.split() + [tmp_path]
            if subprocess.run(["which", cmd[0]], capture_output=True).returncode == 0:
                subprocess.run(cmd, check=False)
                break
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return True
    except Exception:
        return False


def speak_espeak(text: str) -> None:
    """Fallback: espeak-ng with a less robotic voice variant."""
    subprocess.run(
        ["espeak-ng", "-v", "en+m3", "-p", "60", "-s", "160", "-a", "180", text],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SolasAI TTS helper")
    parser.add_argument("text", nargs="+", help="Text to speak")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Edge-TTS voice name")
    args = parser.parse_args()

    full_text = " ".join(args.text).strip()
    if not full_text:
        return

    # Cap text length so it doesn't take too long
    if len(full_text) > 300:
        full_text = full_text[:297] + "..."

    success = asyncio.run(speak_edge(full_text, args.voice))
    if not success:
        speak_espeak(full_text)


if __name__ == "__main__":
    main()
