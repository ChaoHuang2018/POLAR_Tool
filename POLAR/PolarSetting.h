#include <fstream>
#include <iostream>
#include "../nlohmann/json.hpp"

using json = nlohmann::json;
using namespace std;

class PolarSetting
{
public:
    
    unsigned int taylor_order;
    unsigned int bernstein_order;
    unsigned int partition_num;
    int num_threads = 12;
    
    // Berns, Taylor, Mix
    string neuron_approx_type;
    unsigned int neuron_approx;
    
    // Concrete, Symbolic
    string remainder_type;
    bool symb_rem;
    
    //polar setting
    double cutoff_threshold;
    double flowpipe_stepsize;
    unsigned int symbolic_queue_size;
    
    //output setting
    vector<string> output_dim;
    string output_filename;
    
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

    unsigned int get_neuron_approx_type();
        
    void set_remainder_type(string remainder_type);

    unsigned int get_remainder_type();
    
    void set_cutoff_threshold(double cutoff_threshold);
    
    double get_cutoff_threshold();
    
    void set_flowpipe_stepsize(double flowpipe_stepsize);
    
    double get_flowpipe_stepsize();
    
    void set_output_dim(vector<string> output_dim);
    
    vector<string> get_output_dim();
    
    void set_output_filename(string output_filename);

    string get_output_filename();
    
    void set_symbolic_queue_size(unsigned int symbolic_queue_size);

    unsigned int get_symbolic_queue_size();

    int get_num_threads();

    void set_num_threads(int num_threads);

};
