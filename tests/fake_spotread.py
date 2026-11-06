#!/usr/bin/env python3
"""Fake spotread for the pause-behaviour integration test.

Behaves like real spotread's menu: prints the banner, then WAITS for a
key. Every key it receives is appended to a log file with a timestamp.
'\\n' = take a reading (print a minimal reading block + banner again),
'q' = exit.

With FAKE_SELTRIG=<seconds> in the environment it emulates the real
ColorMunki switch-trigger: that long after each menu banner it starts
a reading WITHOUT any key — exactly what spotread does for
switch-equipped instruments while the switch reads closed.
"""
import os
import sys
import time

LOG = sys.argv[1]
SELTRIG = float(os.environ.get("FAKE_SELTRIG", "0"))
BANNER = (
    "Place instrument on spot to be measured,\n"
    "'r' to set reference, 's' to save spectrum,\n"
    "Hit ESC or Q to exit, instrument switch or any other key to take "
    "a reading:\n"
)
READING = (
    "Spectrum from 380.0 to 730.0 nm in 4 steps\n"
    "380.00 0.010 384.00 0.020 388.00 0.030 392.00 0.040\n"
    " Result is XYZ: 0.33354 0.33383 0.33367\n"
)


def log(event):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.time():.2f} {event}\n")


import select  # noqa: E402

sys.stdout.write("Instrument Type:   ColorMunki\n")
sys.stdout.write(BANNER)
sys.stdout.flush()
log("menu")
while True:
    # hardware-switch emulation: trigger a reading without any key
    if SELTRIG and select.select([sys.stdin], [], [], SELTRIG)[0] == []:
        sys.stdout.write(READING)
        sys.stdout.write(BANNER)
        sys.stdout.flush()
        log("hw:spectrum")
        continue
    ch = sys.stdin.read(1)
    if not ch:
        break
    key = "NEWLINE" if ch == "\n" else repr(ch)
    log("key:" + key)
    if ch in ("q", "Q", "\x1b"):
        break
    if ch == "\n":
        sys.stdout.write(READING)
        sys.stdout.write(BANNER)
        sys.stdout.flush()
