import re


def test_cmake_uses_mneme_compilers(build_test_program):
    cache_file = build_test_program["build_dir"] / "CMakeCache.txt"
    print(cache_file)
    contents = cache_file.read_text()
    cc = re.escape(build_test_program["cc"])
    cxx = re.escape(build_test_program["cxx"])

    assert re.search(
        rf"CMAKE_C_COMPILER:[A-Z]+={cc}", contents
    ), "C compiler not set correctly"

    assert re.search(
        rf"CMAKE_CXX_COMPILER:[A-Z]+={cxx}", contents
    ), "C++ compiler not set correctly"
