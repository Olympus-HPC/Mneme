import json
import subprocess as sp
import glob
import sys
from pathlib import Path


for fn in glob.glob("./*.json"):
    with open(fn, "r") as fd:
        rr_data = json.load(fd)
    print("DemangledName:", rr_data["DemangledName"])
    print("NumModules:", len(rr_data["Modules"]))
    print("NumInstances:", len(rr_data["instances"]))
    for k, instance in rr_data["instances"].items():
        print(
            "BlockDims:({0}, {1}, {2})".format(
                instance["BlockDims"]["x"],
                instance["BlockDims"]["y"],
                instance["BlockDims"]["z"],
            )
        )
        print(
            "GridDims:({0}, {1}, {2})".format(
                instance["GridDims"]["x"],
                instance["GridDims"]["y"],
                instance["GridDims"]["z"],
            )
        )
        if not Path(instance["Prologue"]).exists():
            print("Expected prologue file to exist")
            sys.exit(-1)
        if not Path(instance["Epilogue"]).exists():
            print("Expected epilogue file to exist")
            sys.exit(-1)

        cmd = [
            "@MNEME_BIN_FILE@",
            "--mneme-replay-hash",
            str(k),
            "--mneme-replay-json",
            fn,
            "--repeats",
            "1",
        ]

        print(" ".join(cmd))
        res = sp.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        print(res.stderr)
        if res.returncode != 0:
            sys.exit(res.returncode)


sys.exit(0)
