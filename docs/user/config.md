# Runtime Configuration

Mneme exposes a number of possible runtime configuration options, accessible
through environment variables that are applicable either at record time or at replay time.

## Record Time Environment variables


| Environment Variable             | Values                | Description                                                                                                                      |
| -------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| MNEME\_PAGE\_SIZE                | Integer describing GB | The size (in GB) to allocate in the device virtual address space when not specified it defaults to the size of the device memory | 
| MNEME\_LOG\_LEVEL                | The log level by default set to info | Log level of Mneme can take the values of (trace, debug, info, warn, error, critical, off)                        | 
