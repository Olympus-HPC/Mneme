# C++ Development

- **C++ API reference (Doxygen):** [Open the generated API docs](./html/index.html)

If the link 404s locally, ensure Doxygen output is generated into `docs/dev/index/`
before running `mkdocs build` or `mkdocs serve`.

---

## Annotation API (`mneme/MnemeAnnotation.hpp`)

Mneme ships a lightweight C++ header that lets application code attach
verification metadata to device pointers.  The metadata is recorded
alongside the device memory snapshot and consumed during replay to
perform tolerance-aware comparison.

### Include and link

```cpp
#include "mneme/MnemeAnnotation.hpp"
```

The `add_mneme(target)` CMake function (provided by
`MnemeFunctions.cmake`) links the annotation runtime library
(`mnemert`) automatically — no extra `target_link_libraries` call is
needed.

### Quick example

```cpp
double *d_buf = nullptr;
cudaMalloc(&d_buf, N * sizeof(double));

// Typed overload: deduces BuiltinDType from the pointer type.
mneme::annotate<double>(d_buf, mneme::Metadata{
    .threshold      = 1e-6,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::Linf,
    .tag            = std::string("my_buffer"),
});
```

See **[Usage → Verification](../usage/verification.md)** for the full
field reference and end-to-end examples.


---

## Snapshot file format (`mneme/MnemeSnapshotFormat.hpp`)

Prologue and epilogue snapshots share one container: a 16-byte header
followed by a payload whose layout depends on the header.

| Offset | Size | Field                                         |
|--------|------|-----------------------------------------------|
| 0      | 8    | Magic, the ASCII bytes `MNEMESNP`             |
| 8      | 4    | `SnapshotKind` (`Bytes = 1`, `Diff = 2`)      |
| 12     | 4    | Layout version of the payload for that kind   |

`SnapshotHeader::parse` also recognizes two legacy prefixes from before the
container existed: files starting with `MNEME_DIFF_V1` are treated as
`Diff` version 1, and files with no magic at all are treated as `Bytes`
version 0. Those mappings are frozen history; do not change them.

### How versions are owned

Each on-disk layout is decoded by exactly one reader class, and that class
owns its `(kind, version)` pair through a `Layout` member:

```cpp
template <DeviceVendors VendorTypes>
class DiffReaderV1 : public SnapshotReader<VendorTypes> {
public:
  static constexpr SnapshotHeader Layout{SnapshotKind::Diff, 1};
  ...
};
```

Everything else refers to that member rather than repeating the number:

- `SnapshotFormatRegistry` maps the parsed header to a reader by walking a
  table built from `entry<ReaderT>()` rows, one per reader class.
- The writer that produces a layout returns the matching reader's `Layout`
  from `header()`. `FormatWriter::write` emits that header before calling
  `writePayload`, so the stamp on disk can only ever name a layout that has
  a decoder.

### When to bump the version

Bump the version whenever an existing reader would misinterpret the new
bytes: reordering or resizing fields, changing an encoding, or adding a
field that is not self-delimiting. Do not bump for changes that leave the
byte stream identical.

Never edit an existing reader to accept a changed layout. Old recordings
must keep opening, and `ctest -R Serialize` exercises that.

### How to bump the version

Using a diff layout change as the example:

1. Add `DiffReaderV2` next to `DiffReaderV1` with
   `static constexpr SnapshotHeader Layout{SnapshotKind::Diff, 2};` and a
   `read()` that decodes the new payload. Leave `DiffReaderV1` in place.
2. Add `entry<DiffReaderV2<VendorTypes>>()` to the table in
   `SnapshotFormatRegistry::table`.
3. Change `DiffWriter::header()` to return
   `DiffReaderV2<VendorTypes>::Layout`, and update `DiffWriter::writePayload`
   to emit the new layout.
4. Extend `tests/unit_tests/ReadWriteSnapshot.cpp` so the round trip covers
   the new layout, and keep a fixture or hand-built buffer in the previous
   layout so the old reader stays covered.

The version literal appears in exactly one place, the new reader's
`Layout`. Skipping step 2 is the one mistake the compiler cannot catch: the
writer stamps version 2, the registry has no row for it, and opening the
snapshot fails with `Unsupported Mneme snapshot format`.
