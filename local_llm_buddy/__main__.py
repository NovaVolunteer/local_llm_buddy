"""Entry point for ``python -m local_llm_buddy`` (CLI helper)."""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "local_llm_buddy – use 'streamlit run app.py' to start the UI, or "
        "import the package in your own script."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
