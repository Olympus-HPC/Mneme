import json
import glob
import sys
from pathlib import Path


for fn in glob.glob("./*.json"):
    with open(fn, "r") as fd:
        rr_data = json.load(fd)
    print("DemangledName:", rr_data["DemangledName"])
    print("NumModules:", rr_data["NumModules"])
    print("NumInstances:", len(rr_data["instances"]))
    for instance in rr_data["instances"]:
        print(
            "BlockDims:({}, {}, {})",
            instance["BlockDims"]["x"],
            instance["BlockDims"]["y"],
            instance["BlockDims"]["z"],
        )
        print(
            "GridDims:({}, {}, {})",
            instance["GridDims"]["x"],
            instance["GridDims"]["y"],
            instance["GridDims"]["z"],
        )
        if (!Path(instance["prologue"]).exists()):
            print("Expected prologue file to exist")
            return -1
        if (!Path(instance["epilogue"]).exists()):
            print("Expected epilogue file to exist")
            return -1

    print(rr_data)

sys.exit(-1)
