#include "../../POLAR/Polar.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
    System s("./system_benchmark_1.json");
    Specification spec("./specification_benchmark_1.json");
    PolarSetting ps("./polarsetting_benchmark_1.json");
    
    nncs_reachability(s, spec, ps);
}
