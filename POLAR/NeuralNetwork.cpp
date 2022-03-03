#include <fstream>
#include <iostream>
#include "NeuralNetwork.h"

Layer::Layer()
{
}

Layer::Layer(string act, Matrix<Real> w, Matrix<Real> b)
{
    activation = act;
    weight = w;
    bias = b;
}

void Layer::pre_activate(TaylorModelVec<Real> &result, TaylorModelVec<Real> &input, const std::vector<Interval> &domain) const
{
    result.clear();
    result = weight * input;
    Matrix<Real> bias_temp = bias;
    for (int i = 0; i < bias_temp.rows(); i++)
    {
        Polynomial<Real> poly_temp(bias_temp[i][0], domain.size());
        result.tms[i].expansion += poly_temp;
    }
}

void Layer::post_activate(TaylorModelVec<Real> &result, TaylorModelVec<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const
{
    result.clear();
    
    vector<Neuron> neuron_list;
    for (unsigned int i = 0; i < input.tms.size(); ++i)
    {
        Neuron neuron(this->activation);
        neuron_list.push_back(neuron);
    }
    
    for (unsigned int i = 0; i < input.tms.size(); ++i)
    {
        cout << "------"
             << "Neuron " << i << " -------" << endl;
        TaylorModel<Real> tmTemp;
        cout << "Input remainder: " << input.tms[i].remainder << endl;
        neuron_list[i].taylor_model_approx(tmTemp, input.tms[i], domain, polar_setting, setting);
        result.tms.push_back(tmTemp);
    }
}

NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::NeuralNetwork(string filename)
{
    std::ifstream input(filename);
    std::string line;

    // Parse the structure of neural networks
    if (getline(input, line))
    {
    }
    else
    {
        cout << "failed to read file" << endl;
    }
    try
    {
        num_of_inputs = stoi(line);
    }
    catch (std::invalid_argument &e)
    {
        cout << "Problem during string/integer conversion!" << endl;
        cout << line << endl;
    }
    getline(input, line);
    num_of_outputs = stoi(line);
    getline(input, line);
//    try
//    {
//        input >> num_of_hidden_layers;
//    }
//    catch (...)
//    {
//        cout << "aaa" << endl;
//    }
     num_of_hidden_layers = stoi(line);

//    cout << "num_of_inputs" << num_of_inputs << ", " << num_of_outputs << ", " << num_of_hidden_layers << endl;
//
//    exit(0);

    std::vector<int> network_structure(num_of_hidden_layers + 1, 0);
    for (int idx = 0; idx < num_of_hidden_layers; idx++)
    {
        getline(input, line);
        network_structure[idx] = stoi(line);
    }
    network_structure[network_structure.size() - 1] = num_of_outputs;

    // parse the activation function
    std::vector<std::string> activation;
    for (int idx = 0; idx < num_of_hidden_layers + 1; idx++)
    {
        getline(input, line);
        activation.push_back(line);
    }

    // Parse the input text file and store weights and bias

    // a question here. need to confirm on 10/20/2020 afternoon.
    // compute parameters of the input layer
    Matrix<Real> weight0(network_structure[0], num_of_inputs);
    Matrix<Real> bias0(network_structure[0], 1);
    for (int i = 0; i < network_structure[0]; i++)
    {
        for (int j = 0; j < num_of_inputs; j++)
        {
            getline(input, line);
            weight0[i][j] = stod(line);
        }
        getline(input, line);
        bias0[i][0] = stod(line);
    }
    Layer input_layer(activation[0], weight0, bias0);
    // cout << "weight0: " << weight0 << endl;
    // cout << "bias0: " << bias0 << endl;
    layers.push_back(input_layer);

    // compute the parameters of hidden layers
    for (int layer_idx = 0; layer_idx < num_of_hidden_layers; layer_idx++)
    {
        Matrix<Real> weight(network_structure[layer_idx + 1], network_structure[layer_idx]);
        Matrix<Real> bias(network_structure[layer_idx + 1], 1);

        for (int i = 0; i < network_structure[layer_idx + 1]; i++)
        {
            for (int j = 0; j < network_structure[layer_idx]; j++)
            {
                getline(input, line);
                weight[i][j] = stod(line);
            }
            getline(input, line);
            bias[i][0] = stod(line);
        }

        // cout << "weight_" + to_string(layer_idx + 1) + ":" << weight << endl;
        // cout << "bias_" + to_string(layer_idx + 1) + ":" << bias << endl;
        Layer hidden_layer(activation[layer_idx + 1], weight, bias);
        layers.push_back(hidden_layer);
    }
    // Affine mapping of the output
    getline(input, line);
    offset = stod(line);
    
    getline(input, line);
    scale_factor = stod(line);
}

void NeuralNetwork::get_output_tmv(TaylorModelVec<Real> &result, TaylorModelVec<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const
{
    time_t start_timer;
    time_t end_timer;
    double seconds;

    vector<TaylorModelVec<Real>> tmv_all_layer;
    tmv_all_layer.push_back(input);
    for (int s = 0; s < num_of_hidden_layers + 1; s++)
    {
        //cout << "num_of_hidden_layersL: " << num_of_hidden_layers << endl;
        cout << "------------- Layer " << s << " starts. -------------" << endl;
        Layer layer = layers[s];
        
        TaylorModelVec<Real> tmvTemp_pre;
        layer.pre_activate(tmvTemp_pre, tmv_all_layer[s], domain);
        
        
        TaylorModelVec<Real> tmvTemp_post;
        layer.post_activate(tmvTemp_post, tmvTemp_pre, domain, polar_setting, setting);

        tmv_all_layer.push_back(tmvTemp_post);
        
    }

    // cout << "size: " << tmv_all_layer.size() << endl;
    result = tmv_all_layer.back();
    
    Variables vars;
    int x0_id = vars.declareVar("x0");
    int x1_id = vars.declareVar("x1");
    int x2_id = vars.declareVar("x2");
    int x3_id = vars.declareVar("x3");
    int x4_id = vars.declareVar("x4");
    int x5_id = vars.declareVar("x5");
    int u0_id = vars.declareVar("u0");
    int u1_id = vars.declareVar("u1");
    int u2_id = vars.declareVar("u2");
    cout << "----------Before scale and offset: ----------" << endl;
    cout << "output taylor 0: " << endl;
    result.tms[0].output(cout, vars);
    cout << endl;
    cout << "output taylor 1: " << endl;
    result.tms[1].output(cout, vars);
    cout << endl;
    cout << "output taylor 2: " << endl;
    result.tms[2].output(cout, vars);
    cout << endl;
    cout << "--------------------" << endl;

    Matrix<Real> offset_vector(num_of_outputs, 1);
    for (int i = 0; i < num_of_outputs; i++)
    {
        offset_vector[i][0] = -offset;
    }
    // cout << "offset: " << offset << endl;
    result += offset_vector;

    Matrix<Real> scalar(num_of_outputs, num_of_outputs);
    for (int i = 0; i < num_of_outputs; i++)
    {
        scalar[i][i] = scale_factor;
    }
    // cout << "scalar: " << scalar << endl;
    result = scalar * result;
    // cout << "1111111111111111111111" << endl;
    tmv_all_layer.push_back(result);

    Interval box;
    result.tms[0].intEval(box, domain);
    cout << "neural network output range by TMP: " << box << endl;
}

void NeuralNetwork::get_output_tmv_symbolic(TaylorModelVec<Real> &tmv_output, TaylorModelVec<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const
{
    Interval intUnit(-1, 1);

    TaylorModelVec<Real> tmv_layer_input = input;

    Flowpipe fp_layer_input(tmv_layer_input, domain, setting.tm_setting.cutoff_threshold);


    Symbolic_Remainder symbolic_remainder(fp_layer_input, 0);

/*
    int numOfCtl = domain.size() - input.tms.size();

    for(int i=0; i<numOfCtl; ++i)
    {
        symbolic_remainder.polynomial_of_initial_set.pop_back();
    }
*/

    unsigned int numOfLayers = num_of_hidden_layers + 1;

    unsigned int layer_input_dim = fp_layer_input.tmv.tms.size();// - numOfCtl;


    for (unsigned int K = 0; true; ++K)
    {
        cout << "------------- Layer " << K << " starts. -------------" << endl;

        Flowpipe fp_layer_output;

        // evaluate the the initial set x0
        TaylorModelVec<Real> tmv_of_x0;

        if (K == numOfLayers)
        {
            tmv_of_x0 = fp_layer_input.tmvPre;
        }
        else
        {
            Layer layer = layers[K];
            Matrix<Real> weight = layer.get_weight();

 //           cout << weight.rows() << "\t" << weight.cols() << endl;
 //           cout << fp_layer_input.tmvPre.tms.size() << endl;
            tmv_of_x0 = weight * fp_layer_input.tmvPre;
        }

        // the center point of x0's polynomial part
        std::vector<Real> const_of_x0;
        tmv_of_x0.constant(const_of_x0);

        unsigned int rangeDim = tmv_of_x0.tms.size();

        for (unsigned int j = 0; j < rangeDim; ++j)
        {
            Real c;
            tmv_of_x0.tms[j].remainder.remove_midpoint(c);
            const_of_x0[j] += c;

            if (K < numOfLayers)
            {
                Layer layer = layers[K];
                Matrix<Real> bias = layer.get_bias();

                const_of_x0[j] += bias[j][0];
            }
        }

        // introduce a new variable r0 such that x0 = c0 + A*r0, then r0 is origin-centered
        tmv_of_x0.rmConstant();

        // decompose the linear and nonlinear part
        TaylorModelVec<Real> x0_linear, x0_other;
        tmv_of_x0.decompose(x0_linear, x0_other);

        Matrix<Real> Phi_L_i(rangeDim, layer_input_dim);

        x0_linear.linearCoefficients(Phi_L_i);
//cout << Phi_L_i.cols() << "\t" << symbolic_remainder.scalars.size() << endl;
        Matrix<Real> local_trans_linear = Phi_L_i;
        Phi_L_i.right_scale_assign(symbolic_remainder.scalars);
        // compute the remainder part under the linear transformation
        Matrix<Interval> J_i(rangeDim, 1);

        for (unsigned int i = 0; i < symbolic_remainder.Phi_L.size(); ++i)
        {
            symbolic_remainder.Phi_L[i] = Phi_L_i * symbolic_remainder.Phi_L[i];
        }

        symbolic_remainder.Phi_L.push_back(Phi_L_i);

        for (unsigned int i = 1; i < symbolic_remainder.Phi_L.size(); ++i)
        {
            J_i += symbolic_remainder.Phi_L[i] * symbolic_remainder.J[i - 1];
        }

        Matrix<Interval> J_ip1(rangeDim, 1);

        std::vector<Interval> range_of_x0;

        // compute the local initial set
        if (symbolic_remainder.J.size() > 0)
        {
 //            cout << symbolic_remainder.Phi_L[0].cols() << endl;
 //            cout << symbolic_remainder.polynomial_of_initial_set.size() << endl;
            // compute the polynomial part under the linear transformation

            std::vector<Polynomial<Real>> initial_linear = symbolic_remainder.Phi_L[0] * symbolic_remainder.polynomial_of_initial_set;

            // compute the other part
            std::vector<Interval> tmvPolyRange;
            fp_layer_input.tmv.polyRange(tmvPolyRange, fp_layer_input.domain);
            x0_other.insert_ctrunc(fp_layer_output.tmv, fp_layer_input.tmv, tmvPolyRange, fp_layer_input.domain, polar_setting.get_taylor_order(), setting.tm_setting.cutoff_threshold);

            fp_layer_output.tmv.Remainder(J_ip1);

            Matrix<Interval> x0_rem(rangeDim, 1);
            tmv_of_x0.Remainder(x0_rem);
            J_ip1 += x0_rem;

            for (int i = 0; i < rangeDim; ++i)
            {
                fp_layer_output.tmv.tms[i].expansion += initial_linear[i];
            }

            for (int i = 0; i < rangeDim; ++i)
            {
                fp_layer_output.tmv.tms[i].remainder = J_ip1[i][0] + J_i[i][0];
            }

            fp_layer_output.tmv.intEval(range_of_x0, fp_layer_input.domain);

        }
        else
        {
            std::vector<Interval> tmvPolyRange;
            fp_layer_input.tmv.polyRange(tmvPolyRange, fp_layer_input.domain);
            tmv_of_x0.insert_ctrunc(fp_layer_output.tmv, fp_layer_input.tmv, tmvPolyRange, fp_layer_input.domain, polar_setting.get_taylor_order(), setting.tm_setting.cutoff_threshold);

            fp_layer_output.tmv.intEval(range_of_x0, fp_layer_input.domain);

            fp_layer_output.tmv.Remainder(J_ip1);
        }

        symbolic_remainder.J.push_back(J_ip1);

        if (K == numOfLayers)
        {
            tmv_output = fp_layer_output.tmv;

            for (int i = 0; i < tmv_output.tms.size(); ++i)
            {
                tmv_output.tms[i] += const_of_x0[i];
            }

            break;
        }

        // Compute the scaling matrix S.
        std::vector<Real> S, invS;

        if (symbolic_remainder.scalars.size() != rangeDim)
        {
            symbolic_remainder.scalars.resize(rangeDim, 0);
        }

        for (int i = 0; i < rangeDim; ++i)
        {
            Real sup;
            range_of_x0[i].mag(sup);

            if (sup == 0)
            {
                S.push_back(0);
                invS.push_back(1);
                symbolic_remainder.scalars[i] = 0;
            }
            else
            {
                S.push_back(sup);
                Real tmp = 1 / sup;
                invS.push_back(tmp);
                symbolic_remainder.scalars[i] = tmp;
                // range_of_x0[i] = intUnit;
            }
        }

        fp_layer_output.tmv.scale_assign(invS);

        for (int i = 0; i < rangeDim; ++i)
        {
            range_of_x0[i] += const_of_x0[i];
        }

        // Computing a TM overapproximation of the activation function sigmoid(c + z) such that z in I
        std::vector<Interval> newDomain;
        TaylorModelVec<Real> tmv_simple_layer_input(range_of_x0, newDomain);


        Layer layer = layers[K];
        // tmv_simple_layer_input.activate(fp_layer_output.tmvPre, newDomain, layer.get_activation(), ti.order, ti.bernstein_order, ti.partition_num, ti.cutoff_threshold, ti.g_setting, 2);

        layer.post_activate(fp_layer_output.tmvPre,tmv_simple_layer_input, newDomain, polar_setting, setting);

        fp_layer_input.tmv = fp_layer_output.tmv;
        fp_layer_input.tmvPre = fp_layer_output.tmvPre;

        layer_input_dim = layer.get_weight().rows();


    }

    // cout << tmv_output.tms.size() << endl;

    Matrix<Real> offset_vector(num_of_outputs, 1);
    for (int i = 0; i < num_of_outputs; i++)
    {
        offset_vector[i][0] = -1.0 * offset;
    }
    //cout << tmv_output.tms.size() << endl;
    //cout << offset.rows() << endl;

    tmv_output += offset_vector;

    Matrix<Real> scalar(num_of_outputs, num_of_outputs);
    for (int i = 0; i < num_of_outputs; i++)
    {
        scalar[i][i] = scale_factor;
    }
    tmv_output = scalar * tmv_output;

    Interval box;
    tmv_output.tms[0].intEval(box, domain);
    cout << "neural network output range by TMP: " << box << endl;
}


//void NNTaylor::NN_Reach(TaylorModelVec<Real> &tmv_output, TaylorModelVec<Real> &tmv_input, TaylorInfo ti, vector<Interval> &tmv_domain)
//{
//    Interval intUnit(-1, 1);
//
//    TaylorModelVec<Real> tmv_layer_input = tmv_input;
//
//    Flowpipe fp_layer_input(tmv_layer_input, tmv_domain, ti.cutoff_threshold);
//
//    Symbolic_Remainder symbolic_remainder(fp_layer_input);
//
//    unsigned int numOfLayers = nn.get_num_of_hidden_layers() + 1;
//
//    unsigned int layer_input_dim = tmv_domain.size() - 1;
//
//    for (unsigned int K = 0; true; ++K)
//    {
//        cout << "------------- Layer " << K << " starts. -------------" << endl;
//
//        Flowpipe fp_layer_output;
//
//        // evaluate the the initial set x0
//        TaylorModelVec<Real> tmv_of_x0;
//
//        if (K == numOfLayers)
//        {
//            tmv_of_x0 = fp_layer_input.tmvPre;
//        }
//        else
//        {
//            Layer layer = this->nn.get_layers()[K];
//            Matrix<Interval> weight = layer.get_weight();
//
//            Matrix<Real> weight_value(weight.rows(), weight.cols());
//            for (int i = 0; i < weight.rows(); i++)
//            {
//                for (int j = 0; j < weight.cols(); j++)
//                {
//                    weight_value[i][j] = weight[i][j].sup();
//                }
//            }
//
//            //cout << weight_value << endl;
//            //cout << fp_layer_input.tmvPre.tms.size() << endl;
//            tmv_of_x0 = weight_value * fp_layer_input.tmvPre;
//        }
//
//        // the center point of x0's polynomial part
//        std::vector<Real> const_of_x0;
//        tmv_of_x0.constant(const_of_x0);
//
//        unsigned int rangeDim = tmv_of_x0.tms.size();
//        unsigned int rangeDimExt = rangeDim + 1;
//
//        for (unsigned int j = 0; j < rangeDim; ++j)
//        {
//            Real c;
//            tmv_of_x0.tms[j].remainder.remove_midpoint(c);
//            const_of_x0[j] += c;
//
//            if (K < numOfLayers)
//            {
//                Layer layer = this->nn.get_layers()[K];
//                Matrix<Interval> bias = layer.get_bias();
//
//                Matrix<Real> bias_value(bias.rows(), bias.cols());
//                for (int i = 0; i < bias.rows(); i++)
//                {
//                    for (int j = 0; j < bias.cols(); j++)
//                    {
//                        bias_value[i][j] = bias[i][j].sup();
//                    }
//                }
//
//                const_of_x0[j] += bias_value[j][0];
//            }
//        }
//
//        // introduce a new variable r0 such that x0 = c0 + A*r0, then r0 is origin-centered
//        tmv_of_x0.rmConstant();
//
//        // decompose the linear and nonlinear part
//        TaylorModelVec<Real> x0_linear, x0_other;
//        tmv_of_x0.decompose(x0_linear, x0_other);
//
//        Matrix<Real> Phi_L_i(rangeDim, layer_input_dim);
//
//        x0_linear.linearCoefficients(Phi_L_i);
//
//        Matrix<Real> local_trans_linear = Phi_L_i;
//        Phi_L_i.right_scale_assign(symbolic_remainder.scalars);
//        // compute the remainder part under the linear transformation
//        Matrix<Interval> J_i(rangeDim, 1);
//
//        for (unsigned int i = 0; i < symbolic_remainder.Phi_L.size(); ++i)
//        {
//            symbolic_remainder.Phi_L[i] = Phi_L_i * symbolic_remainder.Phi_L[i];
//        }
//
//        symbolic_remainder.Phi_L.push_back(Phi_L_i);
//
//        for (unsigned int i = 1; i < symbolic_remainder.Phi_L.size(); ++i)
//        {
//            J_i += symbolic_remainder.Phi_L[i] * symbolic_remainder.J[i - 1];
//        }
//
//        Matrix<Interval> J_ip1(rangeDim, 1);
//
//        std::vector<Interval> range_of_x0;
//
//        // compute the local initial set
//        if (symbolic_remainder.J.size() > 0)
//        {
//            // cout << symbolic_remainder.Phi_L[0].cols() << endl;
//            // cout << symbolic_remainder.polynomial_of_initial_set.size() << endl;
//            // compute the polynomial part under the linear transformation
//            std::vector<Polynomial<Real>> initial_linear = symbolic_remainder.Phi_L[0] * symbolic_remainder.polynomial_of_initial_set;
//
//            // compute the other part
//            std::vector<Interval> tmvPolyRange;
//            fp_layer_input.tmv.polyRange(tmvPolyRange, fp_layer_input.domain);
//            x0_other.insert_ctrunc(fp_layer_output.tmv, fp_layer_input.tmv, tmvPolyRange, fp_layer_input.domain, ti.order, ti.cutoff_threshold);
//
//            fp_layer_output.tmv.Remainder(J_ip1);
//
//            Matrix<Interval> x0_rem(rangeDim, 1);
//            tmv_of_x0.Remainder(x0_rem);
//            J_ip1 += x0_rem;
//
//            for (int i = 0; i < rangeDim; ++i)
//            {
//                fp_layer_output.tmv.tms[i].expansion += initial_linear[i];
//            }
//
//            for (int i = 0; i < rangeDim; ++i)
//            {
//                fp_layer_output.tmv.tms[i].remainder = J_ip1[i][0] + J_i[i][0];
//            }
//
//            fp_layer_output.tmv.intEval(range_of_x0, fp_layer_input.domain);
//        }
//        else
//        {
//            std::vector<Interval> tmvPolyRange;
//            fp_layer_input.tmv.polyRange(tmvPolyRange, fp_layer_input.domain);
//            tmv_of_x0.insert_ctrunc(fp_layer_output.tmv, fp_layer_input.tmv, tmvPolyRange, fp_layer_input.domain, ti.order, ti.cutoff_threshold);
//
//            fp_layer_output.tmv.intEval(range_of_x0, fp_layer_input.domain);
//
//            fp_layer_output.tmv.Remainder(J_ip1);
//        }
//
//        symbolic_remainder.J.push_back(J_ip1);
//
//        if (K == numOfLayers)
//        {
//            tmv_output = fp_layer_output.tmv;
//
//            for (int i = 0; i < tmv_output.tms.size(); ++i)
//            {
//                tmv_output.tms[i] += const_of_x0[i];
//            }
//
//            break;
//        }
//
//        // Compute the scaling matrix S.
//        std::vector<Real> S, invS;
//
//        if (symbolic_remainder.scalars.size() != rangeDim)
//        {
//            symbolic_remainder.scalars.resize(rangeDim, 0);
//        }
//
//        for (int i = 0; i < rangeDim; ++i)
//        {
//            Real sup;
//            range_of_x0[i].mag(sup);
//
//            if (sup == 0)
//            {
//                S.push_back(0);
//                invS.push_back(1);
//                symbolic_remainder.scalars[i] = 0;
//            }
//            else
//            {
//                S.push_back(sup);
//                Real tmp = 1 / sup;
//                invS.push_back(tmp);
//                symbolic_remainder.scalars[i] = tmp;
//                // range_of_x0[i] = intUnit;
//            }
//        }
//
//        fp_layer_output.tmv.scale_assign(invS);
//
//        for (int i = 0; i < rangeDim; ++i)
//        {
//            range_of_x0[i] += const_of_x0[i];
//        }
//
//        // Computing a TM overapproximation of the activation function sigmoid(c + z) such that z in I
//        std::vector<Interval> newDomain;
//        TaylorModelVec<Real> tmv_simple_layer_input(range_of_x0, newDomain);
//
//        //        tmv_simple_layer_input.output(std::cout, stateVars, tmVars);
//        //        std::cout << std::endl << std::endl;
//        Layer layer = this->nn.get_layers()[K];
//        tmv_simple_layer_input.activate(fp_layer_output.tmvPre, newDomain, layer.get_activation(), ti.order, ti.bernstein_order, ti.partition_num, ti.cutoff_threshold, ti.g_setting, 2);
//
//        //        tmv_simple_layer_output.output(std::cout, stateVars, tmVars);
//        //        std::cout << std::endl << std::endl;
//
//        fp_layer_input.tmv = fp_layer_output.tmv;
//        fp_layer_input.tmvPre = fp_layer_output.tmvPre;
//
//        layer_input_dim = layer.get_neuron_number_this_layer();
//    }
//
//    // cout << tmv_output.tms.size() << endl;
//
//    Matrix<Real> offset(nn.get_num_of_outputs(), 1);
//    for (int i = 0; i < nn.get_num_of_outputs(); i++)
//    {
//        offset[i][0] = -nn.get_offset().sup();
//    }
//    //cout << tmv_output.tms.size() << endl;
//    //cout << offset.rows() << endl;
//
//    tmv_output += offset;
//
//    Matrix<Real> scalar(nn.get_num_of_outputs(), nn.get_num_of_outputs());
//    for (int i = 0; i < nn.get_num_of_outputs(); i++)
//    {
//        scalar[i][i] = nn.get_scale_factor().sup();
//    }
//    tmv_output = scalar * tmv_output;
//    // cout << "1111111111111111111111" << endl;
//
//    Interval box;
//    tmv_output.tms[0].intEval(box, tmv_domain);
//    cout << "neural network output range by TMP_symbolic_remainder: " << box << endl;
//}
