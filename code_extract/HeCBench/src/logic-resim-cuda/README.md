# GPU Accelerated Logic Re-simulation
This is the 3rd place solution to the [ICCAD 2020 CAD contest problem C](http://iccad-contest.org/2020/), which performs timing-aware gate-level logic simulation on GPU, imporves the efficiency comparing to the the past implemetation on large designs. 

This task is seperated into two parts: (1) Graph Preprocessing and (2) GPU Simulation. First (1) generates a intermediate file by input the gate level netlist design with corresponding timing info and behavior library, then (2) can perform simulation base on the previous file and another input testbench contains waveforms of the primary and pseudo-primary inputs of the design. The output is a SAIF file which contains the time nets were of value 0, 1, x, or z for nets in the design for the duration of the specified timestamps.

### 1 Build Simulator
```bash
$ cd Simulation
$ make ARCH=sm_70
```
For more detail please refer to [`Simulation/Makefile`](Simulation/Makefile).

## 2. How to Run
```
Simulator requires five inputs:
- **intermediate**: Intermediate file generated in the previous step.
- **input.vcd**: Contains waveforms of the primary and pseudo-primary inputs of the design.
- **dumpon_time**, **dumpoff_time**: The specified timestamps(ps) for the duration of dumped saif file.
- **result.saif**: Name of the output file.

```bash
$ cd Simulation
$ ./main <intermediate> <input.vcd> <dumpon_time> <dumpoff_time> [result.saif]
```

***Note**: Following the contest instruction 
1. SDF file should only include the ABSOLUTE and IOPATH keywords for consideration.
2. VCD file were reformatted for the consistency sake. (Please refer to [QA38](http://iccad-contest.org/2020/Problem_C/Problem%20C_QA_0928.pdf))
For more information, please refer to [Problem Introduction documentation](http://iccad-contest.org/2020/Problem_C/ICCAD2020_ContestProblemSpecification_ProblemC_08102020.pdf) of the contest.

## 3. Modules
**Simulation**:
- `src`: C++ source code
    - `gate`: database for gates in designs, which stores timing and behaviour information of the cell
    - `parser`: parse the intermediate file and waveforms from testbench file(.vcd), and generate output file(.saif)
    - `sim`: perform simulation on gpu
    - `wave`: database for waveforms in testbench and simulation result
    - `util`: some utility code
- `toys`: toy test cases, the smallest testbench `NV_NVDLA_partition_o_dc_24x33x55_5x5x55x25_int8` provided by contest.

