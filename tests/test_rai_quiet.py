"""The output filter must drain even when a native caller retains the Python GIL."""
from pathlib import Path
import subprocess
import sys


def test_native_gil_held_write_does_not_deadlock_and_preserves_diagnostics():
    code = r'''
import ctypes
import os
import assembly_planner.rai_quiet as rai_quiet

rai_quiet.install()
# PyDLL deliberately retains the GIL, as the RAI binding does during a native solve.
write = ctypes.PyDLL(None).write
write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
write.restype = ctypes.c_ssize_t
spam = b"pairCollision.cpp:libccd: WARNING: ignored\n" * 10000
payload = spam + b"REAL WARNING must survive\n"
buffer = ctypes.create_string_buffer(payload)
assert write(1, buffer, len(payload)) == len(payload)
os.write(2, b"REAL stderr traceback must survive\n")
rai_quiet.uninstall()
assert rai_quiet.dropped() == 10000, rai_quiet.dropped()
# Installation and restoration are reusable and idempotent.
rai_quiet.install()
rai_quiet.install()
os.write(1, b"second run\n")
rai_quiet.uninstall()
rai_quiet.uninstall()
print("restored")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=15, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    assert result.stdout == "REAL WARNING must survive\nsecond run\nrestored\n"
    assert result.stderr == "REAL stderr traceback must survive\n"
