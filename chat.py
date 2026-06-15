"""Compatibility entrypoint for the text-to-voice chat loop."""

from app.cli.chat import main
from app.core.dialogue import *  # noqa: F401,F403 - preserve the old import surface


if __name__ == "__main__":
    main()
