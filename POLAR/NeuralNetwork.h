#include "Neuron.h"
#include <thread>
#include <mutex>
#include <string>
#include <vector>
#include <istream>
#include <cstring>

#include <nlohmann/json.hpp>

#ifdef USE_ONNX_PROTO
  #include <onnx/onnx_pb.h>      
  #include <google/protobuf/io/zero_copy_stream_impl.h>
  #include <google/protobuf/io/coded_stream.h>
  #include <google/protobuf/io/zero_copy_stream_impl_lite.h>
  #include <google/protobuf/text_format.h>
  #include <fstream>
#endif

using json = nlohmann::json;
using namespace flowstar;
using namespace std;

static std::string tolower_str(std::string s){
    for (auto &c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
static bool ends_with(const std::string& s, const std::string& suf){
    if (s.size() < suf.size()) return false;
    return std::equal(suf.rbegin(), suf.rend(), s.rbegin(),
                      [](char a, char b){ return (char)std::tolower((unsigned char)a)==(char)std::tolower((unsigned char)b); });
}
static std::string map_activation_json_to_internal(const std::string& act_json){
    auto a = tolower_str(act_json);
    if (a == "relu")    return "ReLU";      
    if (a == "tanh")    return "tanh";
    if (a == "sigmoid") return "sigmoid";
    if (a == "linear" || a == "none" || a == "affine") return "Affine"; 
    return "Affine";
}

class Layer
{
public:
    // activation of this layer: can be 'ReLU' or 'tanh' or 'sigmoid'  or 'Affine' (linear)
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

// Parse neural network and layer from a text/json file as classes
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
    
    private:
    // detailed implementation 
    // text reader
    void loadFromTxtStream(std::istream& input);

    // json reader
    void loadFromJsonObject(const nlohmann::json& j);

    // onnx reader (future)
    void loadFromOnnxFile(const std::string& path);
};
