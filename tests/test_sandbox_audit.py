"""Offline checks of the read-audit parser on synthetic strace output."""

from ds_research_agent.sandbox.audit import AuditResult, observe, split_wrapper_output


def wrapper(*traces: list[str], end: bool = True) -> str:
    out = ["=== exit 0", "=== elapsed_ns 1000"]
    for i, lines in enumerate(traces):
        out += [f"=== file /audit/t.{100 + i}", *lines]
    if end:
        out.append("=== end")
    return "\n".join(out) + "\n"


def audit(text: str, timed_out: bool = False) -> AuditResult:
    return observe(split_wrapper_output(text), timed_out=timed_out)


OPEN_A = 'openat(AT_FDCWD</scratch>, "/data/a.csv", O_RDONLY|O_CLOEXEC) = 3</data/a.csv>'
EXIT = "+++ exited with 0 +++"


def test_reads_dirs_and_failures_across_processes() -> None:
    a = audit(
        wrapper(
            [OPEN_A, 'openat(AT_FDCWD</scratch>, "/data", O_RDONLY|O_DIRECTORY) = 4</data>', EXIT],
            [
                'openat(AT_FDCWD</scratch>, "/data/b.csv", O_RDONLY) = -1 ENOENT (No such file)',
                'openat(AT_FDCWD</scratch>, "/usr/lib/x.so", O_RDONLY) = 3</usr/lib/x.so>',
                'execve("/data/run.sh", ["/data/run.sh"], 0xffff /* 8 vars */) = 0',
                EXIT,
            ],
        )
    )
    assert a.complete and a.issues == []
    assert a.processes == 2
    assert a.data_reads == ["/data/a.csv", "/data/run.sh"]
    assert a.data_dirs_listed == ["/data"]
    assert len(a.failed_data_opens) == 1


def test_symlink_resolution_uses_decorated_path() -> None:
    line = 'openat(AT_FDCWD</scratch>, "/scratch/link", O_RDONLY) = 3</data/a.csv>'
    assert audit(wrapper([line])).data_reads == ["/data/a.csv"]


def test_escaped_paths_are_decoded() -> None:
    line = (
        'openat(AT_FDCWD</scratch>, "/data/caf\\303\\251.csv", O_RDONLY) = 3'
        "</data/caf\\303\\251.csv>"
    )
    assert audit(wrapper([line])).data_reads == ["/data/café.csv"]


def test_data_prefix_is_a_path_component() -> None:
    line = 'openat(AT_FDCWD</>, "/database/x", O_RDONLY) = 3</database/x>'
    assert audit(wrapper([line])).data_reads == []


def test_missing_end_marker_is_incomplete() -> None:
    a = audit(wrapper([OPEN_A], end=False))
    assert not a.complete and "no end marker" in a.issues[0]


def test_no_processes_is_incomplete() -> None:
    assert not audit(wrapper()).complete


def test_timeout_is_incomplete() -> None:
    a = audit(wrapper([OPEN_A]), timed_out=True)
    assert not a.complete and a.data_reads == ["/data/a.csv"]


def test_undecorated_fd_is_incomplete() -> None:
    a = audit(wrapper(['openat(AT_FDCWD, "/data/a.csv", O_RDONLY) = 3']))
    assert not a.complete and "undecorated" in a.issues[0]


def test_unparseable_open_is_incomplete() -> None:
    a = audit(wrapper(['openat(AT_FDCWD</scratch>, "/data/a.csv", O_RDONLY) = ? <unavailable>']))
    assert not a.complete and "unparsed open" in a.issues[0]


def test_untraced_clone_succeeded_is_incomplete() -> None:
    a = audit(wrapper(["clone(child_stack=NULL, flags=CLONE_UNTRACED|SIGCHLD) = 42"]))
    assert not a.complete and "untraced child" in a.issues[0]


def test_untraced_clone_refused_is_recorded_not_incomplete() -> None:
    a = audit(
        wrapper(
            ["clone(child_stack=NULL, flags=CLONE_UNTRACED|SIGCHLD) = -1 EPERM (Not permitted)"]
        )
    )
    assert a.complete and len(a.blocked_escapes) == 1


def test_io_uring_success_is_incomplete() -> None:
    assert not audit(wrapper(["io_uring_setup(8, 0xffff) = 5<anon_inode:[io_uring]>"])).complete
    refused = audit(wrapper(["io_uring_setup(8, 0xffff) = -1 EPERM (Operation not permitted)"]))
    assert refused.complete and refused.blocked_escapes
