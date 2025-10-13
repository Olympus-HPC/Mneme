#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
HCC        ?= hipcc
OPTIMIZE  = yes
DEBUG     ?= no
LAUNCHER  =
Version ?= default

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(Version), mneme)
  CFLAGS += -fpass-plugin=${MNEME_PREFIX}/lib64/libregdeviceir.so \
            -fno-discard-value-names -ftrivial-auto-var-init=zero
  LDFLAGS += -lmneme_shallow -Wl,-rpath,${MNEME_PREFIX}/lib64/ -L ${MNEME_PREFIX}/lib64/
else ifeq ($(Version), proteus)
CFLAGS += -DENABLE_PROTEUS -I${MNEME_PREFIX}/include \
    -fpass-plugin=${MNEME_PREFIX}/lib64/libProteusPass.so

LDFLAGS = -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
	-L${MNEME_PREFIX}/lib64 \
	-Wl,-rpath,${MNEME_PREFIX}/lib64 \
	-llldCommon -llldELF -lproteus


LDFLAGS += -L${MNEME_PREFIX}/lib64 \
    -Wl,-rpath,${MNEME_PREFIX}/lib64 \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
    -Wl,-rpath,${ROCM_PATH}/llvm/lib \
	-L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
    -lproteus
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(HCC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(HCC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -n 8192

