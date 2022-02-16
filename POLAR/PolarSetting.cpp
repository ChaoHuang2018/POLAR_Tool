#include "PolarSetting.h"

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
