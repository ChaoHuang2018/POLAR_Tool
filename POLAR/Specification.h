#include "../flowstar/flowstar-toolbox/Continuous.h"
#include "../nlohmann/json.hpp"

using json = nlohmann::json;
using namespace flowstar;
using namespace std;

vector<string> split(const std::string &text, char delim);

class Specification
{

    //
public:
    vector<Interval> init;
    int time_steps;
    vector<Interval> unsafe;
    vector<string> safe_set;

public:
    Specification();
    Specification(string filename);
};
