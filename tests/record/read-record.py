import json
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


sys.exit(-1)
