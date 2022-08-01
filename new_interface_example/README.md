# New Interface of POLAR
There are two ways to use POLAR. One is to use POLAR as a library for further development, while anther is to directly use POLAR to verify an NNCS.

## POLAR as a library.

#### Compile POLAR
- Go to the POLAR directory. Assume you are in the current directory /new_interface_example. Then you can use the following command.
```
cd ../POLAR
```
- Compile POLAR as a library
```
make polar_lib
```
#### Use POLAR library: Example Usage
Now you are able to include the APIs of POLAR for your tool. Take benchmark1 in the current directory as an example.
- Go to the directory of the example
```
cd ./benchmark1
```
- Construct a cpp file for the example benchmark_1.cpp
```C++
//include polar.h to use the APIs
#include "../../POLAR/Polar.h"

int main(int argc, char *argv[])
{
    // load json files of system, specification and setting.
    System s("./system_benchmark_1.json");
    Specification spec("./specification_benchmark_1.json");
    PolarSetting ps("./polarsetting_benchmark_1.json");
    
    // verify the system
    nncs_reachability(s, spec, ps);
}
```
- Compile it.
```
make
```
- Run it.
```
./benchmark1
```


## POLAR as a tool.
- Go to the POLAR directory. Assume you are in the current directory /new_interface_example. Then you can use the following command.
```
cd ../POLAR
```
- Compile POLAR as a tool
```
make
```
#### Use POLAR tool: Example Usage
Now you are able to directly use POLAR as a tool. Take benchmark1 in the current directory as an example.
- Go to the directory of the example
```
cd ./benchmark1
```
- Run the following command
```
../../POLAR/polar_tool ./system_benchmark_1.json ./specification_benchmark_1.json ./polarsetting_benchmark_1.json
```
