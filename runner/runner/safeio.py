"""Atomic, Windows-safe file replacement: the temporary file is closed before the rename, has a unique name
per write (no clashes between threads), and the rename is retried when antivirus, indexing or sync tools
briefly lock the target (WinError 5/32). A state file that still cannot be replaced is written directly."""
import json, os, pickle, threading, time
_n = [0]; _lk = threading.Lock()

def _tmp(path):
    with _lk:
        _n[0] += 1; k = _n[0]
    return f"{path}.{os.getpid()}.{threading.get_ident()}.{k}.tmp"

def _replace(tmp, path):
    for i in range(20):
        try:
            os.replace(tmp, path); return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    try:                                         # last resort: write in place
        with open(tmp, "rb") as s, open(path, "wb") as d:
            d.write(s.read())
    finally:
        try: os.remove(tmp)
        except OSError: pass

def write_json(path, obj, **kw):
    tmp = _tmp(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)
    _replace(tmp, path)

def write_pickle(path, obj):
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    _replace(tmp, path)
