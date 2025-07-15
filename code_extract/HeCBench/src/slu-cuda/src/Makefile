#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
VERIFY    = yes
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================
SLU_PATH = .

SLU_LIBS = $(SLU_PATH)/nicslu/lib/nicslu.a \
           $(SLU_PATH)/nicslu/util/nicslu_util.a

program = main

obj = preprocess.o Timer.o numeric.o symbolic.o main.o $(SLU_LIBS)

#===============================================================================
# Sets Flags
#===============================================================================
SLU_INC = -I$(SLU_PATH)/../include \
          -I$(SLU_PATH)/nicslu/include \
          -I$(SLU_PATH)/nicslu/util

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Xcompiler -Wall $(SLU_INC)

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFY
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

Timer.o: $(SLU_PATH)/Timer.cpp
	$(CC) $(CFLAGS) -c $< -o $@

numeric.o: numeric.cu
	$(CC) $(CFLAGS) -c $< -o $@

symbolic.o: $(SLU_PATH)/symbolic.cpp
	$(CC) $(CFLAGS) -c $< -o $@

main.o: $(SLU_PATH)/main.cpp
	$(CC) $(CFLAGS) -c $< -o $@

preprocess.o: $(SLU_PATH)/preprocess.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(SLU_PATH)/nicslu/lib/nicslu.a:
	make -C $(SLU_PATH)/nicslu/

$(SLU_PATH)/nicslu/util/nicslu_util.a:
	make -C $(SLU_PATH)/nicslu/

clean:
	rm -rf $(program) $(obj) *.dat
	make -C $(SLU_PATH)/nicslu/ clean

test: $(program)
	$(LAUNCHER) ./$(program) -i $(SLU_PATH)/nicslu/test/add32.mtx

run: $(program)
	$(LAUNCHER) ./$(program) -i $(SLU_PATH)/nicslu/demo/ASIC_100k.mtx
