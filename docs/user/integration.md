# Integration

To integrate MNeme into your application, you must modify your build to include the Mneme LLVM plugin pass.

This is done by adding Mneme's plugin pass to Clang compilation and its
include directory:
```shell
CXXFLAGS += -fpass-plugin=<install-path>/lib64/libregdeviceir.so 
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


