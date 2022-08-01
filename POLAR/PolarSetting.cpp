#include "PolarSetting.h"

using json = nlohmann::json;
using namespace std;

int PolarSetting::validate(){
    if ((neuron_approx_type != "Berns") && (neuron_approx_type != "Taylor") && (neuron_approx_type != "Mix")) {
        return 0;
    }
    if ((remainder_type != "Concrete") && (remainder_type != "Symbolic")) {
        return 0;
    }
    return 1;
}

PolarSetting::PolarSetting(){
    
}

PolarSetting::PolarSetting(const unsigned int taylor_order, const unsigned int bernstein_order, const unsigned int partition_num, string neuron_approx_type, string remainder_type) {
    this->taylor_order = taylor_order;
    this->bernstein_order = bernstein_order;
    this->partition_num = partition_num;
    this->neuron_approx_type = neuron_approx_type;
    this->remainder_type = remainder_type;
    
    if (validate() == 0) {
        cout << "Wrong neuron approximation type or wrong Taylor model remainder type!" << endl;
        abort();
    }
}

PolarSetting::PolarSetting(string filename)
{
    cout << "Load the setting of POLAR..." << endl;
    ifstream input(filename);
    
    if (filename.substr(filename.find_last_of(".") + 1) == "json")
    {
        // Parse json
        json j = json::parse(input);
        
        taylor_order = j["POLAR_setting"]["taylor_order"];
        bernstein_order = j["POLAR_setting"]["bernstein_order"];
        partition_num = j["POLAR_setting"]["partition_num"];
        neuron_approx_type = j["POLAR_setting"]["neuron_approx_type"];
        remainder_type = j["POLAR_setting"]["remainder_type"];
        
        cutoff_threshold = j["flowstar_setting"]["cutoff_threshold"];
        flowpipe_stepsize = j["flowstar_setting"]["flowpipe_stepsize"];
        
        output_dim = j["output_setting"]["output_dimension"];
        output_filename = j["output_setting"]["output_filename"];
        cout << "Succeed." << endl;
    }
    else
    {
        string line;

        // Parse the structure of neural networks
        if (getline(input, line))
        {
        }
        else
        {
            cout << "failed to read file: Polar Setting" << endl;
        }
        try
        {
            taylor_order = stoi(line);
        }
        catch (invalid_argument &e)
        {
            cout << "Problem during string/integer conversion!" << endl;
            cout << line << endl;
        }
        
        getline(input, line);
        bernstein_order = stoi(line);
        
        getline(input, line);
        partition_num = stoi(line);
        
        getline(input, line);
        neuron_approx_type = line;
        
        getline(input, line);
        remainder_type = line;
    }
}

void PolarSetting::set_taylor_order(unsigned int taylor_order) {
    this->taylor_order = taylor_order;
}

unsigned int PolarSetting::get_taylor_order() {
    return this->taylor_order;
}

void PolarSetting::set_bernstein_order(unsigned int bernstein_order) {
    this->bernstein_order = bernstein_order;
}

unsigned int PolarSetting::get_bernstein_order() {
    return this->bernstein_order;
}

void PolarSetting::set_partition_num(unsigned int partition_num) {
    this->partition_num = partition_num;
}

unsigned int PolarSetting::get_partition_num() {
    return this->partition_num;
}

void PolarSetting::set_neuron_approx_type(string neuron_approx_type) {
    this->neuron_approx_type = neuron_approx_type;
}

string PolarSetting::get_neuron_approx_type() {
    return this->neuron_approx_type;
}

void PolarSetting::set_remainder_type(string remainder_type) {
    this->remainder_type = remainder_type;
}

string PolarSetting::get_remainder_type() {
    return this->remainder_type;
}

void PolarSetting::set_cutoff_threshold(double cutoff_threshold)
{
    this->cutoff_threshold = cutoff_threshold;
}

double PolarSetting::get_cutoff_threshold()
{
    return this->cutoff_threshold;
}

void PolarSetting::set_flowpipe_stepsize(double flowpipe_stepsize)
{
    this->flowpipe_stepsize = flowpipe_stepsize;
}

double PolarSetting::get_flowpipe_stepsize()
{
    return this->flowpipe_stepsize;
}

void PolarSetting::set_output_dim(vector<string> output_dim)
{
    this->output_dim = output_dim;
}

vector<string> PolarSetting::get_output_dim()
{
    return this->output_dim;
}

void PolarSetting::set_output_filename(string output_filename)
{
    this->output_filename = output_filename;
}

string PolarSetting::get_output_filename()
{
    return this->output_filename;
}
