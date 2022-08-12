#include "Polar.h"

int main(int argc, char *argv[])
{
    string system_name = argv[1];
    string specification_name = argv[2];
    string setting_name = argv[3];
    
    System s(system_name);
    Specification spec(specification_name);
//    cout << "123:" << spec.safe_set.size() << endl;
    PolarSetting ps(setting_name);
    
    nncs_reachability(s, spec, ps);
}
