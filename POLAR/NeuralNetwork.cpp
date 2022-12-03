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
//        cout << "------" << "Neuron - " << i << " -------" << endl;
        TaylorModel<Real> tmTemp;
//        cout << "Input remainder: " << input.tms[i].remainder << endl;
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
        cout << "failed to read file: Neural Network." << endl;
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
    
//    cout << "number of hidden layers: " << layers.size() << endl;

}

NeuralNetwork::NeuralNetwork(string filename, string PYTHONPATH)
{
    system((PYTHONPATH + " " + "onnx_converter" + " " + filename).c_str());
	std::ifstream input(filename);
    std::string line;

    // Parse the structure of neural networks
    if (getline(input, line))
    {
    }
    else
    {
        cout << "failed to read file: Neural Network." << endl;
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
    
//    cout << "number of hidden layers: " << layers.size() << endl;

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
//        cout << "pre: " << tmvTemp_pre.tms[0].remainder << endl;
        
        
        TaylorModelVec<Real> tmvTemp_post;
        layer.post_activate(tmvTemp_post, tmvTemp_pre, domain, polar_setting, setting);
//        cout << "post: " << tmvTemp_post.tms[0].remainder << endl;

        tmv_all_layer.push_back(tmvTemp_post);
    }

    // cout << "size: " << tmv_all_layer.size() << endl;
    result = tmv_all_layer.back();
//    cout << "result: " << result.tms[0].remainder << endl;

    Matrix<Real> offset_vector(num_of_outputs, 1);
    
    // Variables vars;
    // int x0_id = vars.declareVar("x0");
    // int x1_id = vars.declareVar("x1");
    // int x2_id = vars.declareVar("x2");
    // int x3_id = vars.declareVar("x3");
    // int x4_id = vars.declareVar("x4");
    // int x5_id = vars.declareVar("x5");
    // int u0_id = vars.declareVar("u0");
    // int u1_id = vars.declareVar("u1");
    // int u2_id = vars.declareVar("u2");
    // cout << "----------Before scale and offset: ----------" << endl;
    // cout << "output taylor 0: " << endl;
    // result.tms[0].output(cout, vars);
    // cout << endl;
    // cout << "output taylor 1: " << endl;
    // result.tms[1].output(cout, vars);
    // cout << endl;
    // cout << "output taylor 2: " << endl;
    // result.tms[2].output(cout, vars);
    // cout << endl;
    // cout << "--------------------" << endl;



    for (int i = 0; i < num_of_outputs; i++)
    {
        if (result.tms[i].expansion.terms.size() == 0)
        {
            Polynomial<Real> tmp_poly(-offset, domain.size());
            result.tms[i].expansion = tmp_poly;
        } else
        {
            result.tms[i].expansion -= offset;
        }
    }

    Matrix<Real> scalar(num_of_outputs, num_of_outputs);
    for (int i = 0; i < num_of_outputs; i++)
    {
        scalar[i][i] = scale_factor;
    }
    // cout << "scalar: " << scalar << endl;
    result = scalar * result;

    tmv_all_layer.push_back(result);

//    Interval box;
//    result.tms[0].intEval(box, domain);
//    cout << "neural network output range by TMP: " << box << endl;
}

void NeuralNetwork::get_output_tmv_symbolic(TaylorModelVec<Real> & result, TaylorModelVec<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting)
{
	// array to keep the matrices of the linear transformations
	vector<Matrix<Real> > Q;

	// array to keep the temporary remainders J
	vector<Matrix<Interval> > J;


	Matrix<Interval> I1(input.tms.size(), 1);

	for(int i=0; i<input.tms.size(); ++i)
	{
		I1[i][0] = input.tms[i].remainder;
	}

	int numOfLayers = num_of_hidden_layers + 1;

	TaylorModelVec<Real> q_i = input;

	Matrix<Interval> latest_J;

	for(int k=0; k<numOfLayers; ++k)
	{
		TaylorModelVec<Real> tmv_layer_input;

		// computing the range of the Taylor model
		vector<Interval> input_range;


		tmv_layer_input = layers[k].weight * q_i;

	    for (int j=0; j<layers[k].bias.rows(); j++)
	    {
	        Polynomial<Real> poly_temp(layers[k].bias[j][0], domain.size());
	        tmv_layer_input.tms[j].expansion += poly_temp;
	    }

	    tmv_layer_input.intEval(input_range, domain);

	    TaylorModelVec<Real> tmv_layer_input_precond = tmv_layer_input;


		// obtaining the vector of Bernstein overapproximations for the activation functions in the k-th layer
		vector<UnivariatePolynomial<Real> > Berns_poly(input_range.size());
		vector<Interval> Berns_rem(input_range.size());

		if(layers[k].activation != "Affine")
		{
			if(layers[k].activation == "ReLU")
			{
				for(int j=0; j<input_range.size(); ++j)
				{
	                UnivariatePolynomial<Real> up;
	                gen_bern_poly(up, layers[k].activation, input_range[j], polar_setting.get_bernstein_order());

	                Berns_poly[j] = up;

	                Real error;
	                up.evaluate(error, Real(0));

	                Interval rem;

	                if(up.coefficients.size() > 2)
	                {
	                	Interval I(-0.5*error, 0.5*error, 1);
	                 	rem = I;
	                 	up.coefficients[0] -= 0.5*error;
//	                 	cout << I << endl;
	                }

	                Berns_rem[j] = rem;
				}
			}
			else
			{
				for(int j=0; j<input_range.size(); ++j)
				{
	                UnivariatePolynomial<Real> up;
	                gen_bern_poly(up, layers[k].activation, input_range[j], polar_setting.get_bernstein_order());

	                Berns_poly[j] = up;

					double error = gen_bern_err_by_sample(Berns_poly[j], layers[k].activation, input_range[j], polar_setting.get_partition_num());

//					cout << error << endl;

//					if(error > 1e-5)
//						cout << error << endl;

					Interval rem(-error, error);
					Berns_rem[j] = rem;
				}
			}
		}





		// extracting the linear part
		Matrix<Real> Q_i(input_range.size());

		if(layers[k].activation != "Affine")
		{
			for(int j=0; j<Berns_poly.size(); ++j)
			{
				if(Berns_poly[j].coefficients.size() > 1)
				{
					Q_i[j][j] = Berns_poly[j].coefficients[1];
				}
				else
				{
					Q_i[j][j] = 0;
				}
			}
		}


		// computing the new TM q_i
		TaylorModelVec<Real> tmvTemp;

		int order = polar_setting.get_bernstein_order();


		if(layers[k].activation != "Affine")
		{
			for(int j=0; j<Berns_poly.size(); ++j)
			{
				TaylorModel<Real> tm_berns;

				if(Berns_poly[j].coefficients.size() > 2)
				{
					TaylorModel<Real> tmTemp(Berns_poly[j].coefficients.back(), domain.size());

					// the nonlinear part
					for (int i = Berns_poly[j].coefficients.size() - 2; i >= 0; --i)
					{
						tmTemp.mul_ctrunc_assign(tmv_layer_input.tms[j], domain, order, setting.tm_setting.cutoff_threshold);

						if(i >= 2)
						{
							TaylorModel<Real> tmTemp2(Berns_poly[j].coefficients[i], domain.size());
							tmTemp += tmTemp2;
						}
					}

					// the linear part
					if(Berns_poly[j].coefficients.size() > 1)
					{
						tmTemp.expansion += tmv_layer_input.tms[j].expansion * Berns_poly[j].coefficients[1];
					}

					// the constant
					if(Berns_poly[j].coefficients.size() > 0)
					{
						Polynomial<Real> polyTemp(Berns_poly[j].coefficients[0], domain.size());
						tmTemp.expansion += polyTemp;
					}

					tm_berns = tmTemp;
				}
				else
				{
					TaylorModel<Real> tmTemp;

					// the linear part
					if(Berns_poly[j].coefficients.size() > 1)
					{
						tmTemp.expansion = tmv_layer_input.tms[j].expansion * Berns_poly[j].coefficients[1];
					}

					// the constant
					if(Berns_poly[j].coefficients.size() > 0)
					{
						Polynomial<Real> polyTemp(Berns_poly[j].coefficients[0], domain.size());
						tmTemp.expansion += polyTemp;
					}

					tm_berns = tmTemp;
				}

				tm_berns.remainder += Berns_rem[j];

				tmvTemp.tms.push_back(tm_berns);
			}
		}
		else
		{
			tmvTemp = tmv_layer_input;

			for(int i=0; i<tmvTemp.tms.size(); ++i)
				tmvTemp.tms[i].remainder = 0;
		}





		// obtaining the vector of Taylor overapproximations for the activation functions in the k-th layer

		vector<UnivariateTaylorModel<Real> > utm_activation(input_range.size());

		interval_utm_setting.order = polar_setting.get_bernstein_order();

		if(layers[k].activation != "Affine" && layers[k].activation != "ReLU")
		{
			if(layers[k].activation == "sigmoid")
			{
				for(int j=0; j<input_range.size(); ++j)
				{
					Real const_part;
					tmv_layer_input_precond.tms[j].constant(const_part);
					tmv_layer_input_precond.tms[j].rmConstant();

					UnivariateTaylorModel<Real> utm_x;
					utm_x.expansion.coefficients.push_back(const_part);
					utm_x.expansion.coefficients.push_back(1);

					Interval x_range = input_range[j] - const_part;

					interval_utm_setting.val = x_range;

					utm_x.sigmoid_taylor(utm_activation[j], x_range,  polar_setting.get_bernstein_order(), setting.g_setting);
				}
			}
			else if(layers[k].activation == "tanh")
			{
				for(int j=0; j<input_range.size(); ++j)
				{
					Real const_part;
					tmv_layer_input_precond.tms[j].constant(const_part);
					tmv_layer_input_precond.tms[j].rmConstant();

					UnivariateTaylorModel<Real> utm_x;
					utm_x.expansion.coefficients.push_back(const_part);
					utm_x.expansion.coefficients.push_back(1);

					Interval x_range = input_range[j] - const_part;

					interval_utm_setting.val = x_range;

					utm_x.tanh_taylor(utm_activation[j], x_range, polar_setting.get_bernstein_order(), setting.g_setting);
				}
			}
		}

		// extracting the linear part
		Matrix<Real> Q_i_Taylor(input_range.size());



		if(layers[k].activation != "Affine" && layers[k].activation != "ReLU")
		{
			for(int j=0; j<utm_activation.size(); ++j)
			{
				if(utm_activation[j].expansion.coefficients.size() > 1)
					Q_i_Taylor[j][j] = utm_activation[j].expansion.coefficients[1];
				else
					Q_i_Taylor[j][j] = 0;
			}
		}



		order = polar_setting.get_taylor_order();

		if(layers[k].activation != "Affine" && layers[k].activation != "ReLU")
		{
			for(int j=0; j<utm_activation.size(); ++j)
			{
				TaylorModel<Real> tm_taylor;

				if(utm_activation[j].expansion.coefficients.size() > 2)
				{
					TaylorModel<Real> tmTemp(utm_activation[j].expansion.coefficients.back(), domain.size());

					// the nonlinear part
					for (int i = utm_activation[j].expansion.coefficients.size() - 2; i >= 0; --i)
					{
						tmTemp.mul_ctrunc_assign(tmv_layer_input_precond.tms[j], domain, order, setting.tm_setting.cutoff_threshold);

						if(i >= 2)
						{
							TaylorModel<Real> tmTemp2(utm_activation[j].expansion.coefficients[i], domain.size());
							tmTemp += tmTemp2;
						}
					}

					// the linear part
					if(utm_activation[j].expansion.coefficients.size() > 1)
					{
						tmTemp.expansion += tmv_layer_input_precond.tms[j].expansion * utm_activation[j].expansion.coefficients[1];
					}

					// the constant
					if(utm_activation[j].expansion.coefficients.size() > 0)
					{
						Polynomial<Real> polyTemp(utm_activation[j].expansion.coefficients[0], domain.size());
						tmTemp.expansion += polyTemp;
					}

					tm_taylor = tmTemp;
				}
				else
				{
					TaylorModel<Real> tmTemp;

					// the linear part
					if(utm_activation[j].expansion.coefficients.size() > 1)
					{
						tmTemp.expansion = tmv_layer_input_precond.tms[j].expansion * utm_activation[j].expansion.coefficients[1];
					}

					// the constant
					if(utm_activation[j].expansion.coefficients.size() > 0)
					{
						Polynomial<Real> polyTemp(utm_activation[j].expansion.coefficients[0], domain.size());
						tmTemp.expansion += polyTemp;
					}

					tm_taylor = tmTemp;
				}

				tm_taylor.remainder += utm_activation[j].remainder;

				if(tm_taylor.remainder.width() < tmvTemp.tms[j].remainder.width())
				{
					tmvTemp.tms[j] = tm_taylor;
					Q_i[j][j] = Q_i_Taylor[j][j];
				}
			}
		}






		Matrix<Real> Phi_i = Q_i * layers[k].weight;



		Matrix<Interval> J_k(tmvTemp.tms.size(), 1);
		tmvTemp.Remainder(J_k);



		// updating Q
		for(int j=0; j<Q.size(); ++j)
		{
			Q[j] = Phi_i * Q[j];
		}

		Q.push_back(Phi_i);


		latest_J = J_k;

		Matrix<Interval> mathbb_J(J_k.rows(), 1);

		if(k > 0)
		{
			mathbb_J = J_k;

			for(int j=1; j<Q.size(); ++j)
			{
				mathbb_J += Q[j] * J[j-1];
			}
		}

		J.push_back(J_k);



		// updating q_i
		q_i = tmvTemp;

		Matrix<Interval> imTemp = Q[0] * I1;

		for(int j=0; j<q_i.tms.size(); ++j)
		{
			q_i.tms[j].remainder += mathbb_J[j][0] + imTemp[j][0];
		}

	}

	result = q_i;


	// applying the scalars and offset
    for(int i = 0; i < result.tms.size(); ++i)
    {
        if(result.tms[i].expansion.terms.size() == 0)
        {
            Polynomial<Real> tmp_poly(-offset, domain.size());
            result.tms[i].expansion = tmp_poly;
        }
        else
        {
        	result.tms[i].expansion -= offset;
        }
    }

    for(int i = 0; i < result.tms.size(); ++i)
    {
    	result.tms[i] *= scale_factor;
    }
}
