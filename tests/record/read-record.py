import glob
import json
import os
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Binary .mneme prologue parser
#
# On-disk format (all integers little-endian, pointer-sized = 8 bytes):
#   char[14] "MNEME_BYTES_V2"
#   uint64  TotalGlobals
#   for each global:
#     uint64 StrLen; char[StrLen] name; uint64 VarSize; void* DevAddr; char[VarSize] data
#   uint64  TotalBlobs
#   for each blob:
#     uint64 ActualSize; uint64 Size; void* DevAddr; char[Size] data
#     uint64 NumAnnotations
#     for each annotation:
#       uint64 Offset; uint64 Extent;
#       Metadata: uint8 builtin; double threshold; uint8 threshold_kind;
#                 uint8 norm; uint64 tag_len; char[tag_len] tag
#   uint64  NumArgs
#   for each arg: uint64 ArgSize; char[ArgSize] data
# ---------------------------------------------------------------------------

def _parse_prologue_metadata(filename):
    """Return a list of annotation dicts for every blob in a prologue file."""
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

    bytes_magic = b"MNEME_BYTES_V2"
    if not data.startswith(bytes_magic):
        raise ValueError("unsupported prologue format")
    off += len(bytes_magic)

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
        for _ in range(u64()):
            offset = u64()
            extent = u64()
            builtin = u8()
            threshold = dbl()
            threshold_kind = u8()
            norm = u8()
            tag_len = u64()
            tag = None
            if tag_len:
                tag = data[off:off + tag_len].decode("utf-8", errors="replace")
                skip(tag_len)
            results.append(
                dict(offset=offset, extent=extent, blob_size=size,
                     builtin=builtin, threshold=threshold,
                     threshold_kind=threshold_kind, norm=norm, tag=tag)
            )
    return results

data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
for fn in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
    with open(fn, "r") as fd:
        rr_data = json.load(fd)
    base_dir = Path(fn).resolve().parent

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
        prologue_path = base_dir / instance["Prologue"]
        epilogue_path = base_dir / instance["Epilogue"]
        if not prologue_path.exists():
            print("Expected prologue file to exist")
            sys.exit(-1)
        if not epilogue_path.exists():
            print("Expected epilogue file to exist")
            sys.exit(-1)

        # Parse the prologue binary to report any non-default blob metadata.
        try:
            mds = _parse_prologue_metadata(prologue_path)
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
                    if md["offset"] == 0 and md["extent"] == md["blob_size"]:
                        print("BlobAnnotation:", " ".join(parts))
                    else:
                        print(
                            "RegionAnnotation:",
                            " ".join(
                                [
                                    f"offset={md['offset']}",
                                    f"extent={md['extent']}",
                                    *parts,
                                ]
                            ),
                        )
        except Exception as e:
            print(f"Warning: could not parse prologue metadata: {e}",
                  file=sys.stderr)


sys.exit(0)
