import re


def test_cmake_uses_mneme_compilers(build_test_program):
    """
    Verify that CMake picked up Mneme's compilers (mneme config cc/cxx)
    when building vecAdd with RDC On/Off.
    """
    cache_file = build_test_program["build_dir"] / "CMakeCache.txt"
    assert cache_file.exists(), f"CMakeCache.txt not found at {cache_file}"

    contents = cache_file.read_text()

    cc = re.escape(build_test_program["cc"])
    cxx = re.escape(build_test_program["cxx"])

    # Example CMakeCache entries:
    #   CMAKE_C_COMPILER:FILEPATH=/path/to/compiler
    #   CMAKE_CXX_COMPILER:FILEPATH=/path/to/compiler++
    assert re.search(
        rf"CMAKE_C_COMPILER:[A-Z]+={cc}", contents
    ), f"C compiler not set correctly.\nFound:\n{contents}"

    assert re.search(
        rf"CMAKE_CXX_COMPILER:[A-Z]+={cxx}", contents
    ), f"C++ compiler not set correctly.\nFound:\n{contents}"
