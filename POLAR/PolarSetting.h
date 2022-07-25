#include <fstream>
#include <iostream>
#include "../nlohmann/json.hpp"

using json = nlohmann::json;
using namespace std;

class PolarSetting
{
protected:
    
    unsigned int taylor_order;
    unsigned int bernstein_order;
    unsigned int partition_num;
    
    // Berns, Taylor, Mix
    string neuron_approx_type;
    
    // Concrete, Symbolic
    string remainder_type;
    
    int validate();
    
public:
    PolarSetting();
    
    PolarSetting(const unsigned int taylor_order, const unsigned int bernstein_order, const unsigned int partition_num, string neuron_approx_type, string remainder_type);
    
    PolarSetting(string filename);

    void set_taylor_order(unsigned int taylor_order);

    unsigned int get_taylor_order();

    void set_bernstein_order(unsigned int bernstein_order);

    unsigned int get_bernstein_order();
        
    void set_partition_num(unsigned int partition_num);

    unsigned int get_partition_num();
        
    void set_neuron_approx_type(string neuron_approx_type);

    string get_neuron_approx_type();
        
    void set_remainder_type(string remainder_type);

    string get_remainder_type();

};
