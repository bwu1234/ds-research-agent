import csv

from ds_research_agent.sandbox.runner import bind_mount_arg


def _parse(arg: str) -> list[str]:
    return next(csv.reader([arg]))


def test_plain_paths_are_unquoted() -> None:
    assert bind_mount_arg("/h/a.csv", "/data/a.csv", readonly=True) == (
        "type=bind,src=/h/a.csv,dst=/data/a.csv,readonly"
    )
    assert bind_mount_arg("/h/o", "/out") == "type=bind,src=/h/o,dst=/out"


def test_comma_in_path_stays_one_field() -> None:
    name = "Report, Part Two, and Other.csv"
    arg = bind_mount_arg(f"/h/{name}", f"/data/legal/{name}", readonly=True)
    assert _parse(arg) == [
        "type=bind",
        f"src=/h/{name}",
        f"dst=/data/legal/{name}",
        "readonly",
    ]
