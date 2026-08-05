# The plugin is built against LLVM_INSTALL_DIR, so an opt from a different LLVM
# cannot load it. NO_DEFAULT_PATH keeps the search off the system paths.
find_program(OPT opt
    PATHS ${LLVM_INSTALL_DIR}/bin ${LLVM_INSTALL_DIR}/libexec/llvm
    NO_DEFAULT_PATH)
if(NOT OPT)
    message(FATAL_ERROR "Pass tests require opt from the LLVM install")
endif()
message(STATUS "Found opt at ${OPT}")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats
import os
import tempfile
import atexit
import shutil

config.name = 'Mneme pass tests'
config.test_format = lit.formats.ShTest(True)
config.environment = os.environ.copy()

config.suffixes = ['.ll']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
exec_root = tempfile.mkdtemp(prefix='lit.tmp.', dir='${CMAKE_CURRENT_BINARY_DIR}')
config.test_exec_root = exec_root
atexit.register(lambda: shutil.rmtree(exec_root, ignore_errors=False))

config.substitutions.append(('%opt', lit_config.params['OPT']))
config.substitutions.append(('%plugin', lit_config.params['PLUGIN']))
config.substitutions.append(('%FILECHECK', lit_config.params['FILECHECK']))
"
)

add_test(NAME purge_autoinit_pass
    COMMAND ${LIT} -a -vv
        -DFILECHECK=${FILECHECK}
        -DOPT=${OPT}
        -DPLUGIN=$<TARGET_FILE:MnemePassPlugin>
        purge-autoinit.ll)
set_tests_properties(purge_autoinit_pass PROPERTIES LABELS "MnemePass")
