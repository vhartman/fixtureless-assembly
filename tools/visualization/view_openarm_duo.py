#!/usr/bin/env python3
"""Interactive viewer for the merged OpenArm duo .g -- edit openarm_duo.g and
press ENTER here to reload (or run with --watch to auto-reload on save)."""
import os
from pathlib import Path
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import robotic as ry
PATH = str(_REPO_ROOT / 'src' / 'rai' / 'openarm' / 'openarm_duo.g')
C = ry.Config()
C.addFile(PATH)
v = C.view()
watch = '--watch' in sys.argv
last = os.path.getmtime(PATH)
print('viewing', PATH, '-- edit the file; press ENTER to reload, Ctrl-C to quit')
try:
    while True:
        if watch:
            m = os.path.getmtime(PATH)
            if m != last:
                last = m
                C.clear()
                C.addFile(PATH)
                print('[reloaded]')
            time.sleep(0.5)
        else:
            input()
            C.clear(); C.addFile(PATH)
            print('[reloaded]')
except (KeyboardInterrupt, EOFError):
    pass
