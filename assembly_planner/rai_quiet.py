"""Drop rai's libccd collision spam at the file descriptor, without touching rai.

    -- pairCollision.cpp:libccd:354(0) WARNING: called MPR penetration for non intersecting...
    -- pairCollision.cpp:libccd:359(0) WARNING: but GJK says intersection

rai prints these from C++ (pairCollision.cpp) straight to fd 1, so `contextlib.redirect_stdout`
and every other Python-level trick is blind to them -- Python never sees the bytes. The only
place to intercept is the DESCRIPTOR, so we splice a pipe onto it and use a small independent
process to forward every line except the matched ones. A Python reader THREAD is unsafe:
native solvers may hold the GIL while writing more than a pipe buffer, deadlocking the writer
against the reader waiting for that same GIL. The filter process imports only the standard
library and does not share the solver's GIL.

Ordering is preserved: Python's own prints go down the same pipe, so they cannot be reordered
relative to rai's.

Usage:
    from assembly_planner import rai_quiet; rai_quiet.install()  # process-wide
    with rai_quiet.quiet():  ...                   # scoped
"""
import atexit
import json
import mmap
import os
import re
import struct
import subprocess
import sys
import tempfile

# Substrings that identify the spam. Deliberately NARROW: this must never eat a real message.
# mp_export's own "[mp-export] WARNING: step N has NO witness" is a genuine diagnostic that a
# blanket WARNING filter would swallow -- that exact mistake hid a real bug for a whole session,
# so match on the C++ SOURCE FILE, not on the word WARNING.
DEFAULT_PATTERNS = (
    "pairCollision.cpp",
    "libccd",
)

_state = {"installed": False, "pumps": [], "saved": {}, "dropped": 0}


def dropped():
  """How many lines have been suppressed so far (0 is a good hint the filter isn't matching)."""
  return _state["dropped"] + sum(struct.unpack("<Q", counter[:8])[0]
                                for _proc, _file, counter in _state["pumps"])


def _pump(read_fd, out_fd, matcher, counter):
  buf = b""
  count = 0

  def write_all(data):
    # os.write is allowed to complete only part of a buffer.
    while data:
      data = data[os.write(out_fd, data):]

  while True:
    try:
      chunk = os.read(read_fd, 65536)
    except OSError:
      break
    if not chunk:
      break
    buf += chunk
    # keep any partial trailing line in the buffer so a split mid-line is not misjudged
    lines = buf.split(b"\n")
    buf = lines.pop()
    out = []
    for ln in lines:
      if matcher(ln):
        count += 1
      else:
        out.append(ln)
    if out:
      try:
        write_all(b"\n".join(out) + b"\n")
      except OSError:
        break
    counter[:8] = struct.pack("<Q", count)
  if buf:                                    # flush whatever was left un-terminated
    try:
      write_all(buf)
    except OSError:
      pass


def install(patterns=DEFAULT_PATTERNS, fds=(1, 2)):
  """Filter `fds` for the rest of the process. Idempotent; restores at interpreter exit."""
  if _state["installed"]:
    return
  for fd in fds:
    try:
      sys.stdout.flush() if fd == 1 else sys.stderr.flush()
    except Exception:
      pass
    saved = os.dup(fd)                       # the REAL terminal, kept for the pump to write to
    r, w = os.pipe()
    counter_file = tempfile.TemporaryFile()
    counter_file.truncate(8)
    counter = mmap.mmap(counter_file.fileno(), 8)
    try:
      proc = subprocess.Popen(
          [sys.executable, os.path.abspath(__file__), "--pump", json.dumps(list(patterns)),
           str(counter_file.fileno())], stdin=r, stdout=saved, stderr=saved,
          pass_fds=(counter_file.fileno(),))
    except BaseException:
      for handle in (r, w, saved):
        os.close(handle)
      counter.close()
      counter_file.close()
      uninstall()
      raise
    os.dup2(w, fd)                           # everything aimed at fd now lands in the pipe
    os.close(w)
    os.close(r)
    _state["pumps"].append((proc, counter_file, counter))
    _state["saved"][fd] = saved
    _state["installed"] = True
  atexit.register(uninstall)


def uninstall():
  """Restore the real fds AND DRAIN. The drain is not optional: without it, whatever is still
  in the pipe when the process exits is lost -- which silently ate a traceback the first time
  this shipped. Restoring the fd drops the last writer, the pump then reads EOF and exits, and
  we join it so its final write lands before the interpreter goes away."""
  if not _state["installed"]:
    return
  # ORDER MATTERS. dup2 first (drops the pipe's last writer, so the pump reads EOF), then JOIN
  # so the pump's final write lands, and only THEN close the saved fd. Closing before the join
  # -- as this first did -- destroys the pump's output target mid-drain, and the tail of the
  # run is lost: it silently swallowed a traceback, which is far worse than the spam.
  for fd, saved in _state["saved"].items():
    try:
      sys.stdout.flush() if fd == 1 else sys.stderr.flush()
    except Exception:
      pass
    os.dup2(saved, fd)                       # last writer gone -> the pump sees EOF
  for proc, counter_file, counter in _state["pumps"]:
    try:
      proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
      proc.terminate()
      proc.wait(timeout=5)
      os.write(2, b"[rai-quiet] filter failed to drain; trailing output may be missing\n")
    _state["dropped"] += struct.unpack("<Q", counter[:8])[0]
    counter.close()
    counter_file.close()
  for _fd, saved in _state["saved"].items():
    try:
      os.close(saved)                        # only now -- the pump is done writing to it
    except OSError:
      pass
  _state["pumps"].clear()
  _state["installed"] = False
  _state["saved"].clear()


class quiet:
  """Scoped form: `with rai_quiet.quiet(): ...`"""

  def __init__(self, patterns=DEFAULT_PATTERNS, fds=(1, 2)):
    self._p, self._f = patterns, fds

  def __enter__(self):
    install(self._p, self._f)
    return self

  def __exit__(self, *exc):
    uninstall()
    return False


def self_test():
  """Verify (a) C-level writes are filtered, (b) real output survives, (c) the WRITER's cost.

  Timing measures the writer only -- that is rai's side, the part on the critical path. The
  pump runs on another thread and only matters if it cannot keep up with the producer. The
  baseline writes to a scratch fd, NOT to stdout, so the benchmark does not pollute the very
  output the test then greps (an earlier version did, and its own baseline spam looked like
  filter leakage).
  """
  import time
  SPAM = b"-- pairCollision.cpp:libccd:354(0) WARNING: called MPR penetration for non int...\n"
  REAL = b"[mp-export] WARNING: step 2 has NO witness -- this MUST survive\n"
  N = 20000

  devnull = os.open(os.devnull, os.O_WRONLY)
  t0 = time.perf_counter()
  for _ in range(N):
    os.write(devnull, SPAM)
  t_null = time.perf_counter() - t0
  os.close(devnull)

  install()
  t0 = time.perf_counter()
  for _ in range(N):
    os.write(1, SPAM)                        # C-level, exactly how rai emits it
  t_write = time.perf_counter() - t0         # writer cost ONLY (pump drains concurrently)
  for _ in range(200):
    os.write(1, REAL)
  print("[self-test] python print() also survives")
  sys.stdout.flush()
  time.sleep(0.3)                            # let the pump drain before restoring the fd
  n_drop = dropped()
  uninstall()
  print(f"[self-test] writer cost for {N} spam lines: "
        f"{t_write*1000:.1f}ms into the filter vs {t_null*1000:.1f}ms into /dev/null "
        f"({t_write/max(t_null,1e-9):.2f}x)")
  print(f"[self-test] dropped={n_drop} (expect {N}); 200 real WARNING lines were kept above")
  print("[self-test] any 'pairCollision' line in THIS output would be leakage")


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--pump":
    patterns = json.loads(sys.argv[2])
    rx = re.compile(b"|".join(re.escape(p.encode()) for p in patterns))
    with mmap.mmap(int(sys.argv[3]), 8) as counter:
      _pump(0, 1, rx.search, counter)
  else:
    self_test()
