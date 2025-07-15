# bb_segsort (segmented sort): Fast Segmented Sort on GPUs

This repository provides a fast segmented sort on NVIDIA GPUs. The library contains many parallel kernels for different types of segments. In particular, the kernels for solving short/medium segments are automatically generated to efficiently utilize registers in GPUs. More details about the kernels and code generation can be found in the original paper.

## Original Work

* [Original GitHub repository](https://github.com/vtsynergy/bb_segsort)
* Contact Email: kaixihou@vt.edu

## Improvements in this fork

* Added key only version
* Asynchronous execution using a single CUDA stream inside bb_segsort_run
* No temporary memory allocation inside bb_segsort_run
* Reduced memory overhead
* Two dimensional kernel grid to avoid index calculations
* Avoiding boundaries check by using one-past-the-end offset
* No dependency on Thrust

## Interface differences

* This version expects two offset arrays, one for begin and one for (one-past-the-)end offsets of the segments
* You can use a single array and pass `offsets` and `offsets+1` if the segments are densly packed (end of a segment is begin of next segment). Be sure to include the last one-past-the-end offset.

## Usage

To use the segmented sort (**bb_segsort**), you need to include the `bb_segsort.cuh` (key-value) or `bb_segsort_keys.cuh` (key only).
Use `bb_segsort(...)` if you don't care about memory allocation or asynchronous execution, or use `bb_segsort_run(...)` and provide your own memory allocation and stream.

Note, bb_segsort utilizes an unstable sorting network as the building block; thus, equivalent elements are not guaranteed to keep the original relative order.

## Example

[main.cu](main.cu) contains an example of how to use (**bb_segsort**). Adapt the Makefile to fit your system. Especially, you may need to change the ARCH according to your GPU platform. For example, if you are using a P100 GPU, you should update ARCH to 61.

Compile using make:

```[Bash]
$ make
```

After compilation, run the executable:

```[Bash]
$ make run
```

## License

Please refer to the included LICENSE file.
