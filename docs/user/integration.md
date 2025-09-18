# Integration with User Codebases

Mneme has three phases, which are detailed below. 

## Building a _recordable_ executable
To integrate MNeme into your application, you must modify your build to include the Mneme LLVM plugin pass.

This is done by adding Mneme's plugin pass to Clang compilation and its
include directory:
```shell
CXXFLAGS += -fpass-plugin=<install-path>/lib64/libregdeviceir.so 
```

Further, `libmneme_shallow` needs to be linked as such:

```
LDFLAGS += -lmneme_shallow -Wl,-rpath,${MNEME_PREFIX}/lib64/ -L ${MNEME_PREFIX}/lib64/
```


```
## Using CMake

To use Mneme with CMake, make sure the Mneme install directory is in
`CMAKE_PREFIX_PATH`, or pass it as `-Dmneme_DIR=<install-path>`.
Then, in your project's `CMakeLists.txt` simply add the following two lines:

```cmake
find_package(mneme REQUIRED)

add_mneme(<target>)
```

Where `target` is the name of your library or executable target.

## Recording a trace from the device and host 

```shell 
LD_PRELOAD=<path-to-Mneme>/build/lib/lib64/librecord.so \
MNEME_PAGE_SIZE=16 \
./application
```

## Replaying and Tuning with `mneme replay|tune`


### Replay example

### Tuning example

### Understanding various tuning paramaters
#### `--specialize`
#### `--prune`
#### `--internalize`

### Debugging tips and Gotchas


