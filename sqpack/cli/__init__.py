"""sqpack CLI dispatcher.

Usage:
    sqpack solve N [flags]
    sqpack refine RESULT.json [flags]
    sqpack render RESULT.json [flags]
    sqpack info RESULT.json
"""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sqpack",
        description="Pack n unit squares into the smallest enclosing square.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    from . import solve as solve_mod
    from . import refine as refine_mod
    from . import render as render_mod
    from . import info as info_mod

    solve_mod.register(subparsers)
    refine_mod.register(subparsers)
    render_mod.register(subparsers)
    info_mod.register(subparsers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
