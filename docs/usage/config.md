# Runtime Configuration

Mneme exposes a number of possible runtime configuration options, accessible
through environment variables that are applicable either at record time or at replay time.


| Environment Variable             | Used during           | Description                                                                                                                      |
| -------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| MNEME\_PAGE\_SIZE                | Recording             | The size (in GB) to allocate in the device virtual address space when not specified it defaults to the size of the device memory | 
| MNEME\_LOG\_LEVEL                | Recording|Replaying   | Log level of Mneme can take the values of (trace, debug, info, warn, error, critical, off)                                       | 
| MNEME\_LOG\_DIR                  | Recording|Replaying   | The directory in which we will store results into                                                                                | 
| MNEME\_RR\_KERNELS               | Recording             | A regex to be used to choose which kernels to recoding. Only kernels with matching names will be recorded                        | 
| MNEME\_MAX\_RECORDINGS               | Recording             | An integer describing the maximum number of recordings we can do on the same kernel.                        | 
