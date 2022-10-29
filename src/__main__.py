"""The marianne entry point script."""
# src/__main__.py

from . import __app_name__, cli


def main():
    cli.app(prog_name=__app_name__)


if __name__ == "__main__":
    main()
