#!/usr/bin/env python3
"""
SolasAI Robot Bridge
====================
Local HTTP server that controls a LEGO Mindstorms Robot Inventor (set 51515)
or SPIKE Prime hub via the MicroPython REPL over USB serial.

How it works
------------
The hub exposes a MicroPython REPL over the USB cable.  This bridge connects
to that serial port, sends one-line `hub.port.*` commands, waits for the
movement to finish, then handles the next command.

TurboWarp / SolasAI sends a ROBOT_CMD token in the AI reply:
    forward:3:50|left:1:30|stop:0:0

The TurboWarp project strips that token and POSTs it here.

Setup
-----
    pip install flask flask-cors pyserial
    python robot_bridge.py           # auto-detects hub port

    # or specify port manually:
    ROBOT_PORT=/dev/ttyACM0 python robot_bridge.py

Command string format
---------------------
Commands are separated by |.  Each command is:
    <action>:<duration_seconds>:<speed_or_freq>

Supported actions:
    forward   – drive forward  (ports A=left, B=right)
    backward  – drive backward
    left      – pivot left in place
    right     – pivot right in place
    spin      – spin on the spot (A forward, B backward)
    stop      – immediately brake both motors
    beep      – play a tone (duration=seconds, speed field=frequency Hz)

Example:
    forward:3:50|left:1:40|beep:0.5:440|stop:0:0

API
---
    GET  /status        → {"connected": bool, "port": str|null}
    POST /connect       → {"port": str?}        → {"ok": bool, "message": str}
    POST /disconnect    → {}
    POST /execute       → {"commands": "…"}     → {"ok": bool, "results": […]}
"""

import os
import time
import threading
import json

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("WARNING: pyserial not installed. Run: pip install pyserial")

from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("WARNING: flask-cors not installed. Run: pip install flask-cors")

# ── Config ────────────────────────────────────────────────────────────────────
BRIDGE_PORT = int(os.environ.get("ROBOT_BRIDGE_PORT", 8900))
ROBOT_PORT = os.environ.get("ROBOT_PORT", "")          # e.g. /dev/ttyACM0 or COM3
MOTOR_LEFT_PORT = os.environ.get("MOTOR_LEFT_PORT", "A")
MOTOR_RIGHT_PORT = os.environ.get("MOTOR_RIGHT_PORT", "B")
INTER_CMD_PAUSE = float(os.environ.get("INTER_CMD_PAUSE", "0.1"))

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app, origins=["http://localhost:*", "https://turbowarp.org", "*"])

hub_serial = None
hub_lock = threading.Lock()

# ── Port detection ────────────────────────────────────────────────────────────
SPIKE_KEYWORDS = ["lego", "spike", "mindstorm", "robot inventor", "51515"]

def find_spike_port():
    """Auto-detect the SPIKE Prime hub USB serial port."""
    if not SERIAL_AVAILABLE:
        return None
    if ROBOT_PORT:
        return ROBOT_PORT
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if any(kw in desc for kw in SPIKE_KEYWORDS):
            return p.device
    # Fallback: first ttyACM / cu.usbmodem device (Linux/Mac)
    for p in serial.tools.list_ports.comports():
        dev = p.device.lower()
        if "ttyacm" in dev or "usbmodem" in dev:
            return p.device
    return None

# ── Hub connection ────────────────────────────────────────────────────────────
def hub_connect(port=None):
    """Open serial connection to hub and clear any running program."""
    global hub_serial
    if not SERIAL_AVAILABLE:
        return False, "pyserial is not installed"

    port = port or find_spike_port()
    if not port:
        return False, (
            "No SPIKE Prime / Robot Inventor hub found. "
            "Connect the hub via USB and make sure no program is running on it. "
            "You can also set ROBOT_PORT env var (e.g. /dev/ttyACM0 or COM4)."
        )
    try:
        s = serial.Serial(port, 115200, timeout=3)
        time.sleep(0.5)
        s.write(b"\x03\r\n")   # Ctrl+C: interrupt any running MicroPython program
        time.sleep(0.4)
        s.read_all()            # flush startup noise
        hub_serial = s
        return True, port
    except Exception as exc:
        return False, str(exc)

def hub_disconnect():
    global hub_serial
    if hub_serial and hub_serial.is_open:
        try:
            hub_serial.close()
        except Exception:
            pass
    hub_serial = None

# ── MicroPython REPL exec ─────────────────────────────────────────────────────
def hub_exec(code: str) -> str:
    """Send one line of MicroPython to the hub REPL and return the output."""
    global hub_serial
    if not hub_serial or not hub_serial.is_open:
        ok, msg = hub_connect()
        if not ok:
            return f"ERROR: {msg}"

    with hub_lock:
        try:
            hub_serial.read_all()                      # flush
            hub_serial.write((code.strip() + "\r\n").encode())
            time.sleep(0.15)
            return hub_serial.read_all().decode(errors="replace").strip()
        except Exception as exc:
            hub_serial = None
            return f"ERROR: {exc}"

# ── Command → MicroPython ─────────────────────────────────────────────────────
def _cmd_to_code(action: str, duration: float, param: int) -> str:
    """
    Convert a parsed command to a hub MicroPython one-liner.

    The hub module API (works on both Robot Inventor 51515 firmware ≥ 1.3 and
    SPIKE Prime) uses:
        hub.port.<X>.motor.run_for_time(milliseconds, speed)
        hub.sound.beep(frequency, milliseconds)
    """
    ms = max(50, int(duration * 1000))
    L = MOTOR_LEFT_PORT.upper()
    R = MOTOR_RIGHT_PORT.upper()
    spd = max(10, min(100, param))

    if action == "forward":
        return (
            f"import hub as _h;"
            f"_h.port.{L}.motor.run_for_time({ms},{spd});"
            f"_h.port.{R}.motor.run_for_time({ms},{spd})"
        )
    if action in ("backward", "back", "reverse"):
        return (
            f"import hub as _h;"
            f"_h.port.{L}.motor.run_for_time({ms},-{spd});"
            f"_h.port.{R}.motor.run_for_time({ms},-{spd})"
        )
    if action == "left":
        return (
            f"import hub as _h;"
            f"_h.port.{L}.motor.run_for_time({ms},-{spd});"
            f"_h.port.{R}.motor.run_for_time({ms},{spd})"
        )
    if action == "right":
        return (
            f"import hub as _h;"
            f"_h.port.{L}.motor.run_for_time({ms},{spd});"
            f"_h.port.{R}.motor.run_for_time({ms},-{spd})"
        )
    if action == "spin":
        return (
            f"import hub as _h;"
            f"_h.port.{L}.motor.run_for_time({ms},{spd});"
            f"_h.port.{R}.motor.run_for_time({ms},-{spd})"
        )
    if action == "stop":
        return (
            f"import hub as _h;"
            f"_h.port.{L}.motor.brake();"
            f"_h.port.{R}.motor.brake()"
        )
    if action == "beep":
        freq = max(44, min(10000, param))
        return f"import hub as _h; _h.sound.beep({freq},{ms})"

    return f"# unknown action: {action}"


def run_commands(command_string: str) -> list:
    """
    Parse and execute a pipe-separated command string.

    Returns a list of result dicts with action, duration, output.
    """
    results = []
    segments = [s.strip() for s in command_string.split("|") if s.strip()]

    for seg in segments:
        parts = seg.split(":")
        action = parts[0].lower().strip() if parts else "stop"
        try:
            duration = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
        except ValueError:
            duration = 1.0
        try:
            param = int(parts[2]) if len(parts) > 2 and parts[2] else 50
        except ValueError:
            param = 50

        code = _cmd_to_code(action, duration, param)
        output = hub_exec(code)
        results.append({
            "action": action,
            "duration": duration,
            "param": param,
            "output": output,
            "ok": not output.startswith("ERROR")
        })

        # Wait for the movement to finish before sending the next command.
        # stop and beep are instantaneous commands; all others need to wait.
        if action not in ("stop",):
            time.sleep(duration + INTER_CMD_PAUSE)

    return results

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/status")
def route_status():
    connected = bool(hub_serial and hub_serial.is_open)
    return jsonify({
        "connected": connected,
        "port": hub_serial.port if connected else None,
        "serial_available": SERIAL_AVAILABLE
    })


@app.post("/connect")
def route_connect():
    data = request.get_json(silent=True) or {}
    port = data.get("port") or None
    ok, msg = hub_connect(port)
    return jsonify({"ok": ok, "message": msg})


@app.post("/disconnect")
def route_disconnect():
    hub_disconnect()
    return jsonify({"ok": True})


@app.post("/execute")
def route_execute():
    data = request.get_json(silent=True) or {}
    commands = str(data.get("commands", "")).strip()
    if not commands:
        return jsonify({"ok": False, "error": "No commands provided"}), 400
    try:
        results = run_commands(commands)
        all_ok = all(r.get("ok", False) for r in results)
        return jsonify({"ok": all_ok, "results": results})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"  SolasAI Robot Bridge  →  http://localhost:{BRIDGE_PORT}")
    print("=" * 60)
    if not SERIAL_AVAILABLE:
        print("  ⚠  pyserial missing — install with:  pip install pyserial")
    else:
        print("  Scanning for SPIKE Prime / Robot Inventor hub…")
        ok, msg = hub_connect()
        if ok:
            print(f"  ✓ Hub connected on {msg}")
        else:
            print(f"  ✗ {msg}")
            print("  Connect the hub via USB then call POST /connect")
    print()
    print("  Endpoints:")
    print(f"    GET  http://localhost:{BRIDGE_PORT}/status")
    print(f"    POST http://localhost:{BRIDGE_PORT}/connect")
    print(f"    POST http://localhost:{BRIDGE_PORT}/execute   {{commands:'forward:3:50|stop:0:0'}}")
    print()
    app.run(host="127.0.0.1", port=BRIDGE_PORT, debug=False)
