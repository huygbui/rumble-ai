"""Compatibility entrypoint for low-latency TTS synthesis."""

from app.cli.say import main
from app.core.speech import *  # noqa: F401,F403 - preserve the old import surface


if __name__ == "__main__":
    main()
