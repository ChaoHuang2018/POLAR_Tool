#include "Neuron.h"
#include <thread>
#include <mutex>

using namespace flowstar;
using namespace std;

class Layer
{
public:
    // activation of this layer: can be 'ReLU' or 'tanh' or 'sigmoid'
    string activation;
    // even though weight and bias are real matrix, we use interval to describe the access of each matrix for convenience
    Matrix<Real> weight;
    Matrix<Real> bias;

public:
    Layer();
    Layer(string act, Matrix<Real> w, Matrix<Real> b);

    string get_activation()
    {
        return this->activation;
    }

    Matrix<Real> get_weight()
    {
        return this->weight;
    }

    Matrix<Real> get_bias()
    {
        return this->bias;
    }
    
    void pre_activate(TaylorModelVec<Real> &result, TaylorModelVec<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting) const;
 
    void post_activate(TaylorModelVec<Real> &result, TaylorModelVec<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const;
};

// Parse neural network and layer from a text file as classes
// Please provide the get and set function for each member in the two classes.

class NeuralNetwork
{

    //
protected:
    int num_of_inputs;
    // current version only support nn with scalar output, i.e., 1-dimesional output
    int num_of_outputs;
    int num_of_hidden_layers;
    // use interval type for offset and scale_factor
    // If needed, please declare the access of each matrix as a double
    Real offset;
    Real scale_factor;
    // include hidden layers and output layer
    vector<Layer> layers;

public:
    NeuralNetwork();
    NeuralNetwork(string filename);
    NeuralNetwork(string filename, string PYTHONPATH);
    int get_num_of_inputs()
    {
        return this->num_of_inputs;
    }

    int get_num_of_outputs()
    {
        return this->num_of_outputs;
    }

    int get_num_of_hidden_layers()
    {
        return this->num_of_hidden_layers;
    }

    Real get_offset()
    {
        return this->offset;
    }

    Real get_scale_factor()
    {
        return this->scale_factor;
    }

    vector<Layer> get_layers()
    {
        return this->layers;
    }
    
    void get_output_tmv(TaylorModelVec<Real> &result, TaylorModelVec<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const;
    
    void get_output_tmv_symbolic(TaylorModelVec<Real> &result, TaylorModelVec<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting);
    
};
