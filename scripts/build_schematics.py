"""Render the ZTE schematics for the paper and dissertation; the same command as `zte-schematics`, kept beside the
other artifact builders so `scripts/` stays the one place to look for them.
"""

from __future__ import annotations

from zte.cli.schematics import main

if __name__ == '__main__':
    main()
