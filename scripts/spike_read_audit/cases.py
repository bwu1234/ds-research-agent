"""Programs run under audit in the read-audit spike (as the unprivileged user).

Each case reads some of /data in a different way. It prints a JSON line
describing what it tried; that output is the program's own claim and is
compared with, never substituted for, the trace.
"""

import ctypes
import json
import mmap
import os
import shutil
import signal
import subprocess
import sys
import time

A, B, GPKG, BIG = "/data/a.csv", "/data/b.csv", "/data/geo.gpkg", "/data/big.csv"
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long


def report(**kw: object) -> None:
    print(json.dumps(kw), flush=True)


def py_open() -> None:
    with open(A) as f:
        report(bytes=len(f.read()))


def pandas_csv() -> None:
    import pandas as pd

    report(rows=len(pd.read_csv(B)))


def gdal_gpkg() -> None:
    import pyogrio.raw

    _meta, _fids, geometry, _fields = pyogrio.raw.read(GPKG)
    report(rows=len(geometry), gdal=pyogrio.__gdal_version_string__)


def child_cat() -> None:
    out = subprocess.run(["cat", A], capture_output=True, check=True).stdout
    report(bytes=len(out))


def child_python() -> None:
    code = f"print(len(open({B!r}).read()))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    report(bytes=int(out.stdout))


def grandchild_shell() -> None:
    out = subprocess.run(["sh", "-c", f"sh -c 'head -c 5 {A}'"], capture_output=True, check=True)
    report(bytes=len(out.stdout))


def symlink_in_scratch() -> None:
    link = "/scratch/link.csv"
    os.symlink(A, link)
    with open(link) as f:
        report(bytes=len(f.read()))


def proc_fd_reopen() -> None:
    fd = os.open(B, os.O_RDONLY)
    with open(f"/proc/self/fd/{fd}") as f:
        report(bytes=len(f.read()))


def mmap_read() -> None:
    with open(A, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        report(bytes=len(m[:]))


def copy_then_read() -> None:
    shutil.copy(A, "/scratch/copy.csv")  # may use copy_file_range/sendfile
    with open("/scratch/copy.csv") as f:
        report(bytes=len(f.read()))


def orphan_after_exit() -> None:
    # A detached grandchild that reads after the main program has exited.
    subprocess.Popen(["sh", "-c", f"sleep 1; cat {B} > /dev/null"], start_new_session=True)
    report(spawned=True)


def list_dir() -> None:
    report(entries=sorted(os.listdir("/data")))


def kill_tracer() -> None:
    tracer = os.getppid()
    try:
        os.kill(tracer, signal.SIGKILL)
        report(killed=tracer)
    except PermissionError as e:
        report(errno=e.errno, tracer=tracer)
    with open(A) as f:
        f.read()


def _raw(nr: int, *args: int) -> tuple[int, int]:
    r = libc.syscall(ctypes.c_long(nr), *(ctypes.c_long(a) for a in args))
    return r, ctypes.get_errno()


# aarch64 syscall numbers (asm-generic table).
SYS_CLONE, SYS_IO_URING_SETUP = 220, 425
SYS_NAME_TO_HANDLE_AT, SYS_OPEN_BY_HANDLE_AT = 264, 265
CLONE_UNTRACED, SIGCHLD = 0x00800000, 17


def clone_untraced() -> None:
    # fork() semantics (no CLONE_VM, stack 0) plus CLONE_UNTRACED, which
    # stops a tracer's automatic attach to the child.
    pid, err = _raw(SYS_CLONE, CLONE_UNTRACED | SIGCHLD, 0, 0, 0, 0)
    if pid == 0:
        with open(B) as f:
            f.read()
        os._exit(0)
    if pid < 0:
        report(errno=err)
        return
    os.waitpid(pid, 0)
    report(child=pid)


def io_uring() -> None:
    params = ctypes.create_string_buffer(120)
    r, err = _raw(SYS_IO_URING_SETUP, 8, ctypes.addressof(params))
    report(result=r, errno=err)
    if r >= 0:
        os.close(r)


def open_by_handle() -> None:
    # A real handle for B, then reopen it by handle (bypasses path lookup).
    # struct file_handle: u32 handle_bytes, i32 handle_type, u8 f_handle[128].
    handle = ctypes.create_string_buffer(8 + 128)
    ctypes.c_uint.from_buffer(handle).value = 128
    mount_id = ctypes.c_int()
    path = ctypes.create_string_buffer(B.encode())
    r, err = _raw(
        SYS_NAME_TO_HANDLE_AT, -100, ctypes.addressof(path), ctypes.addressof(handle),
        ctypes.addressof(mount_id), 0,
    )  # fmt: skip
    if r < 0:
        report(name_to_handle_errno=err)
        return
    mount_fd = os.open("/data", os.O_RDONLY | os.O_DIRECTORY)
    fd, err = _raw(SYS_OPEN_BY_HANDLE_AT, mount_fd, ctypes.addressof(handle), os.O_RDONLY)
    if fd >= 0:
        report(opened=True, bytes=len(os.read(fd, 1 << 20)))
    else:
        report(opened=False, errno=err)


def big_csv() -> None:
    import pandas as pd

    t = time.perf_counter()
    n = len(pd.read_csv(BIG))
    report(rows=n, read_s=time.perf_counter() - t)


def nothing() -> None:
    report(ok=True)


def tracee_caps() -> None:
    with open("/proc/self/status") as f:
        caps = dict(line.split(":\t") for line in f if line.startswith("Cap"))
    report(uid=os.getuid(), **{k: v.strip() for k, v in caps.items()})


def make_gpkg() -> None:
    """Setup only: write a small synthetic GeoPackage to /scratch."""
    import struct

    import numpy as np
    import pyogrio.raw

    pts = [(-111.9 + i / 10, 40.7 + i / 20) for i in range(5)]
    geometry = np.array([struct.pack("<BIdd", 1, 1, x, y) for x, y in pts], dtype=object)
    names = np.array([f"site{i}" for i in range(5)], dtype=object)
    pyogrio.raw.write(
        "/scratch/geo.gpkg",
        geometry,
        [names],
        fields=["name"],
        geometry_type="Point",
        crs="EPSG:4326",
        driver="GPKG",
    )
    report(written=len(pts))


CASES = {
    f.__name__: f
    for f in [
        make_gpkg,
        py_open,
        pandas_csv,
        gdal_gpkg,
        child_cat,
        child_python,
        grandchild_shell,
        symlink_in_scratch,
        proc_fd_reopen,
        mmap_read,
        copy_then_read,
        orphan_after_exit,
        list_dir,
        kill_tracer,
        clone_untraced,
        io_uring,
        open_by_handle,
        big_csv,
        nothing,
        tracee_caps,
    ]
}

if __name__ == "__main__":
    CASES[sys.argv[1]]()
