import glob
import json
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Binary .mneme prologue parser
#
# On-disk format (all integers little-endian, pointer-sized = 8 bytes):
#   uint64  TotalGlobals
#   for each global:
#     uint64 StrLen; char[StrLen] name; uint64 VarSize; void* DevAddr; char[VarSize] data
#   uint64  TotalBlobs
#   for each blob:
#     uint64 ActualSize; uint64 Size; void* DevAddr; char[Size] data
#     Metadata: uint8 builtin; double threshold; uint8 threshold_kind;
#               uint8 norm; uint64 tag_len; char[tag_len] tag
#   uint64  NumArgs
#   for each arg: uint64 ArgSize; char[ArgSize] data
# ---------------------------------------------------------------------------

def _parse_prologue_metadata(filename):
    """Return a list of Metadata dicts for every blob in a prologue file."""
    with open(filename, "rb") as f:
        data = f.read()
    off = 0

    def u64():
        nonlocal off
        v = struct.unpack_from("<Q", data, off)[0]
        off += 8
        return v

    def u8():
        nonlocal off
        v = struct.unpack_from("B", data, off)[0]
        off += 1
        return v

    def dbl():
        nonlocal off
        v = struct.unpack_from("<d", data, off)[0]
        off += 8
        return v

    def skip(n):
        nonlocal off
        off += n

    # globals
    for _ in range(u64()):
        str_len = u64()
        skip(str_len)           # name
        var_size = u64()
        skip(8)                 # dev_addr
        skip(var_size)          # data

    # blobs
    results = []
    for _ in range(u64()):
        u64()                   # actual_size
        size = u64()
        skip(8)                 # dev_addr
        skip(size)              # blob data
        builtin       = u8()
        threshold     = dbl()
        threshold_kind = u8()
        norm          = u8()
        tag_len       = u64()
        tag = None
        if tag_len:
            tag = data[off:off + tag_len].decode("utf-8", errors="replace")
            skip(tag_len)
        results.append(
            dict(builtin=builtin, threshold=threshold,
                 threshold_kind=threshold_kind, norm=norm, tag=tag)
        )
    return results

for fn in glob.glob("./*.json"):
    with open(fn, "r") as fd:
        rr_data = json.load(fd)
    d_name = rr_data["DemangledName"]
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

        # Parse the prologue binary to report any non-default blob metadata.
        try:
            mds = _parse_prologue_metadata(instance["Prologue"])
            for md in mds:
                if (md["threshold"] != 0.0 or md["builtin"] != 0
                        or md["norm"] != 0 or md["threshold_kind"] != 0
                        or md["tag"] is not None):
                    parts = [
                        f"threshold={md['threshold']}",
                        f"threshold_kind={md['threshold_kind']}",
                        f"builtin={md['builtin']}",
                        f"norm={md['norm']}",
                    ]
                    if md["tag"] is not None:
                        parts.append(f"tag={md['tag']}")
                    print("BlobAnnotation:", " ".join(parts))
        except Exception as e:
            print(f"Warning: could not parse prologue metadata: {e}",
                  file=sys.stderr)


sys.exit(0)
