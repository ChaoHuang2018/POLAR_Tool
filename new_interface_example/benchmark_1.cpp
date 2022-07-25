#include "../POLAR/Polar.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
    System s("./dynamics_benchmark_1.json");
    NeuralNetwork nn("nn_1_sigmoid");
    Specification spec("./specification_benchmark_1.json");
    PolarSetting ps("./polarsetting_benchmark_1.json");
    
    nncs_reachability(s, nn, spec, ps);
}
