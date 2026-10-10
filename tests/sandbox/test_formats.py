"""Each format in the benchmark's file-type inventory loads in the sandbox.

Synthetic fixtures only: one sandbox run writes a small file per format into
scratch with the image's own libraries, and a second run reads each one back
as a mounted input, so the read audit covers every reader. The real files
are checked locally by ``scripts/check_sandbox_formats.py``.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from ds_research_agent.sandbox import InputMount, SandboxRunner

pytestmark = pytest.mark.docker

# A public example TLE (the sgp4 package's documentation uses the ISS).
TLE = (
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991\n"
    "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482\n"
)
SP3 = """\
#dP2019  9 13  0  0  0.00000000      2 ORBIT IGS14 FIT  TST
## 2071 432000.00000000    10.00000000 58739 0.0000000000000
+    1   L47  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
*  2019  9 13  0  0  0.00000000
PL47   1234.567890  -2345.678901   6543.210987 999999.999999
*  2019  9 13  0  0 10.00000000
PL47   1240.000000  -2340.000000   6540.000000 999999.999999
EOF
"""

WRITE = f"""
import json, struct
import numpy as np, pandas as pd, pyogrio.raw, cdflib.cdfwrite
S = "/scratch/"
pd.DataFrame({{"a": [1, 2, 3], "b": ["x", "y", "z"]}}).to_csv(S + "t.csv", index=False)
with pd.ExcelWriter(S + "t.xlsx") as w:
    pd.DataFrame({{"a": [1, 2]}}).to_excel(w, sheet_name="one", index=False)
    pd.DataFrame({{"b": [3.5]}}).to_excel(w, sheet_name="two", index=False)
g = np.array([struct.pack("<BIdd", 1, 1, i / 10, i / 20) for i in range(3)], dtype=object)
pyogrio.raw.write(S + "t.gpkg", g, [np.array(list("abc"), dtype=object)], fields=["name"],
                  geometry_type="Point", crs="EPSG:4326", driver="GPKG")
np.savez(S + "t.npz", grid=np.arange(12.0).reshape(3, 4), lat=np.array([1.0, 2.0, 3.0]))
c = cdflib.cdfwrite.CDF(S + "t.cdf", cdf_spec={{"Majority": "row_major"}})
c.write_var({{"Variable": "B", "Data_Type": cdflib.cdfwrite.CDF.CDF_REAL8, "Num_Elements": 1,
             "Rec_Vary": True, "Dim_Sizes": []}}, var_data=np.array([1.5, 2.5, 3.5]))
c.close()
open(S + "t.json", "w").write(json.dumps({{"rows": [{{"k": 1}}, {{"k": 2}}]}}))
open(S + "t.html", "w").write("<table><tr><th>k</th><th>v</th></tr>"
                               "<tr><td>1</td><td>2</td></tr></table>")
open(S + "t.hdr", "w").write("<?xml version='1.0'?><Earth_Explorer_Header><Fixed_Header>"
                              "<File_Name>X</File_Name></Fixed_Header></Earth_Explorer_Header>")
open(S + "t.dat", "w").write("2024   1  0  1.5  2.5\\n2024   1  1  9.9 99.9\\n")
open(S + "t.tle", "w").write({TLE!r})
open(S + "t.sp3", "w").write({SP3!r})
pd.DataFrame({{"a": [1, 2]}}).to_parquet(S + "t.parquet")
"""

READ = """
import json
import cdflib, geopandas, lxml.etree, lxml.html, numpy as np, pandas as pd, pyogrio
from sgp4.api import Satrec, jday
D = "/data/f/t."
out = {}
out["csv"] = pd.read_csv(D + "csv").shape[0]
out["xlsx"] = {k: v.shape[0] for k, v in pd.read_excel(D + "xlsx", sheet_name=None).items()}
out["gpkg_layers"] = [l[0] for l in pyogrio.list_layers(D + "gpkg")]
out["gpkg"] = len(geopandas.read_file(D + "gpkg"))
with np.load(D + "npz", allow_pickle=False) as z:
    out["npz"] = {k: list(z[k].shape) for k in z.files}
cdf = cdflib.CDF(D + "cdf")
out["cdf"] = cdf.varget("B").tolist()
out["json"] = len(json.load(open(D + "json"))["rows"])
out["html"] = pd.read_html(D + "html")[0].shape[0]
out["hdr"] = lxml.etree.parse(D + "hdr").getroot().tag
out["dat"] = pd.read_csv(D + "dat", sep=r"\\s+", header=None).shape[1]
l1, l2 = open(D + "tle").read().splitlines()
e, r, v = Satrec.twoline2rv(l1, l2).sgp4(*jday(2019, 12, 9, 12, 0, 0))
out["tle"] = [e, round(sum(x * x for x in r) ** 0.5)]
out["sp3"] = sum(1 for line in open(D + "sp3") if line.startswith("PL47"))
out["parquet"] = pd.read_parquet(D + "parquet").shape[0]
print(json.dumps(out))
"""

FORMATS = ["csv", "xlsx", "gpkg", "npz", "cdf", "json", "html", "hdr", "dat", "tle", "sp3"]


def test_every_inventoried_format_loads(runner: SandboxRunner) -> None:
    scratch = runner.new_scratch()
    made = runner.run(textwrap.dedent(WRITE), [], scratch=scratch)
    assert made.exit_code == 0, made.stderr
    inputs = [
        InputMount(host_path=scratch / f"t.{ext}", container_path=f"/data/f/t.{ext}")
        for ext in [*FORMATS, "parquet"]
    ]
    r = runner.run(textwrap.dedent(READ), inputs)
    assert r.exit_code == 0, r.stderr
    out = json.loads(r.stdout)
    assert out == {
        "csv": 3,
        "xlsx": {"one": 2, "two": 1},
        "gpkg_layers": ["t"],
        "gpkg": 3,
        "npz": {"grid": [3, 4], "lat": [3]},
        "cdf": [1.5, 2.5, 3.5],
        "json": 2,
        "html": 1,
        "hdr": "Earth_Explorer_Header",
        "dat": 5,
        "tle": [0, out["tle"][1]],
        "sp3": 2,
        "parquet": 2,
    }
    assert 6_500 < out["tle"][1] < 7_000  # km from Earth's centre: low orbit
    assert r.audit.complete
    assert set(r.audit.data_reads) == {i.container_path for i in inputs}


def test_analysis_packages_import(runner: SandboxRunner) -> None:
    r = runner.run(
        "import scipy.stats, sklearn.linear_model, statsmodels.api, pyproj, shapely\n"
        "print(pyproj.CRS.from_epsg(4326).name)",
        [],
    )
    assert r.exit_code == 0, r.stderr
    assert r.stdout.strip() == "WGS 84"
