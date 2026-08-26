#!/usr/bin/env python3
"""Interactive viewer for the merged OpenArm duo .g -- edit openarm_duo.g and
press ENTER here to reload (or run with --watch to auto-reload on save)."""
import sys, time, os
import robotic as ry
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'src', 'rai', 'openarm', 'openarm_duo.g')
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
