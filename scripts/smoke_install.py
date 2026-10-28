"""Start an installed CLI and check health; sends no model requests."""

import argparse
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("executable")
args = parser.parse_args()
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
process = subprocess.Popen(
    [
        args.executable,
        "serve",
        "--upstream-base-url",
        "http://127.0.0.1:1/v1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
try:
    for _ in range(100):
        if process.poll() is not None:
            raise SystemExit("Installed service exited during startup")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/healthz", timeout=0.2
            ) as response:
                assert json.load(response) == {"status": "ok"}
            print("Installed service startup and /healthz passed; no model requests sent")
            break
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    else:
        raise SystemExit("Installed service did not become healthy")
finally:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
