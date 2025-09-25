# Integration with User Codebases

Mneme has three phases, which are detailed below. 

## Phase 1: Building a _recordable_ executable
First, to integrate Mneme in your application, you must modify your application's build to include the Mneme LLVM plugin pass. This can be done as follows with a Clang-based compiler framework, where `MNEME_PREFIX` refers to the installation path for Mneme:

```shell 
CXXFLAGS += -Xclang -disable-O0-optnone \
-fpass-plugin=${MNEME_PREFIX}/lib64/libregdeviceir.so \
-fno-discard-value-names -ftrivial-auto-var-init=zero
```

Further, `libmneme_shallow` needs to be linked as such:

```shell
LDFLAGS += -lmneme_shallow -Wl,-rpath,${MNEME_PREFIX}/lib64/ \
-L ${MNEME_PREFIX}/lib64/
```

To use Mneme with CMake, make sure the Mneme install directory is in
`CMAKE_PREFIX_PATH`, or pass it as `-Dmneme_DIR=<install-path>`. Then, in your project's `CMakeLists.txt` simply add the following two lines, where `target` is the name of your library or executable target:

```cmake
## Using CMake

find_package(mneme REQUIRED)
add_mneme(<target>)
```

## Phase 2: Recording a trace from the GPU device 
The next step involves recording the traces from the GPU device as well as the host. This can be accomplished by using `LD_PRELOAD` to load the recording library, and then executing your application. 

```shell 
LD_PRELOAD=${MNEME_PREFIX}/lib64/librecord.so \
MNEME_PAGE_SIZE=16 \
./application
```

The `MNEME_PAGE_SIZE` variable, specifies the page size in GB, and can be utilized to address any memory related issues, such as out-of-memory errors that might come up in the recording process. Other relevant environment variables can be found [here](config.md).

 Mneme will record all kernel invocations of the application and will store their memory and application code under the current directory. 

TBD. Describe the JSON DB files as and the device, host files being generated here....

## Phase 3: Replaying and Tuning with Mneme

We describe the `replay` tool as well as the `mneme execute|tune` tool in detail on the  [Replaying, Optimizing and Tuning with Mneme](usage.md) page. 



