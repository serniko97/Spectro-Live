"""Entry point: parse options and launch the Qt GUI."""
from __future__ import annotations

import argparse

from . import __version__


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spectro-live",
                                 description="Realtime-style spectrum viewer for "
                                             "spot spectrometers (ColorMunki Photo / ArgyllCMS).")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--source", default=None,
                    choices=("spotread", "simulator"),
                    help="start with this source selected")
    ap.add_argument("--start", action="store_true",
                    help="start measuring immediately")
    args = ap.parse_args(argv)

    # Qt UI (PySide6) — the only GUI: the legacy tkinter fallback was
    # removed (the Qt app is the maintained, tested, themed UI)
    from .qtapp import run
    run(source=args.source, start=args.start)


if __name__ == "__main__":
    main()
