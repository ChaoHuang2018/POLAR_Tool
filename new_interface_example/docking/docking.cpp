#include "../../POLAR/Polar.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
    System s("./system_docking.json");
    Specification spec("./specification_docking.json");
    PolarSetting ps("./polarsetting_docking.json");
    
    nncs_reachability(s, spec, ps);
}
