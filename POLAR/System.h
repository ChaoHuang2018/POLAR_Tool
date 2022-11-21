#include "../flowstar/flowstar-toolbox/Continuous.h"
#include <fstream>
#include "../nlohmann/json.hpp"
#include "NeuralNetwork.h"

using json = nlohmann::json;
using namespace flowstar;
using namespace std;

class System
{

    //
public:
    int num_of_states;
    int num_of_control;
    vector<string> state_name_list;
    vector<string> control_name_list;
    vector<string> ode_list;
    double control_stepsize;
    NeuralNetwork nn;

public:
    System();
    System(string filename);
};
