#include <fstream>
#include <iostream>
#include "NeuralNetwork.h"
#include <thread>
#include <mutex>

Layer::Layer()
{
}

Layer::Layer(string act, Matrix<Real> w, Matrix<Real> b)
{
    activation = act;
    weight = w;
    bias = b;
}
 
void Layer::pre_activate(TaylorModelVec<Real> &result, TaylorModelVec<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting) const
{
    result.clear();
    result = weight * input;
    Matrix<Real> bias_temp = bias;
    
    if (polar_setting.get_num_threads() < 0) 
    {
        for (int i = 0; i < bias_temp.rows(); i++)
        {
            Polynomial<Real> poly_temp(bias_temp[i][0], domain.size());
            result.tms[i].expansion += poly_temp;
        }
    } 
    else 
    {
        result.clear();
        result = weight * input;
        Matrix<Real> bias_temp = bias;

        
        // Get the number of neurons
        int num_rows = bias_temp.rows();
        //cout << "-------- Multi-thread All " << num_rows << " Neurons Pre Activate -------- " << endl;
        // Create placeholder for the biases and expansions
        Real bias_arr[num_rows];
        Polynomial<Real> expansions[num_rows];
        
        // Fillout the placeholders
        for (int i = 0; i < num_rows; i++) 
        {
            bias_arr[i] = bias_temp[i][0];
            expansions[i] = result.tms[i].expansion;
        }

        // Create multiple threads;
        vector<std::thread> threads;
        //vector<thread> ths(this->num_threads);
        // Add biases to the expansions
        for (int i = 0; i < num_rows; i++) 
        {
            threads.emplace_back([&](int j) {
                //mtx.lock();
                Polynomial<Real> poly_temp(bias_arr[j], domain.size());
                expansions[j] += poly_temp;
            }, i);
        }
        for (auto& th: threads) 
        {
            th.join();
        }
        //threads.clear();
        
        // Pass the updated expansions to the Taylor Model
        for (int i = 0; i < num_rows; i++)
        {
            result.tms[i].expansion = expansions[i];
        }
        //cout << "-------- Multi-thread All Neurons Pre Done -------- " << endl;
        //cout << "Result size: " << result.tms.size() << endl;
    }
}

void Layer::post_activate(TaylorModelVec<Real> &result, TaylorModelVec<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const
{
    result.clear();
    

    if (polar_setting.get_num_threads() < 0) 
    {
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
    else 
    {
        int num_rows = input.tms.size();
        //cout << "-------- Multi-thread All " << num_rows << " Neurons Post Activate -------- " << endl;
    
        // Create TaylorModel placeholder arrays for the input and output TaylorModels
        TaylorModel<Real> tm_is[num_rows], tm_os[num_rows];
        // Fill out the input TaylorModels' placeholder arrary
        std::copy(input.tms.begin(), input.tms.end(), tm_is);
        // Create multiple threads
        vector<std::thread> threads;
        //vector<thread> ths(this->num_threads);
        // Fill out the output TaylorModels' placeholder array
        for (unsigned int i = 0; i < num_rows; ++i) 
        {
            threads.emplace_back([&](int j) {
                Neuron neuron(this->activation);
                //mtx.lock();
                neuron.taylor_model_approx(tm_os[j], tm_is[j], domain, polar_setting, setting);
                //mtx.unlock();
            }, i);
        }
        for (auto& th: threads)
        {
            th.join();
        }
        threads.clear();
        // Create result Taylor models
        result.tms.reserve(num_rows);
        
        for (int i = 0; i < num_rows; i++)
        {
            result.tms.push_back(tm_os[i]);
        }
        //cout << "-------- Multi-thread All Neurons Post Done -------- " << endl;
        //cout << "Result size: " << result.tms.size() << endl;
    }
        
}

NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::NeuralNetwork(string filename)
{
    // NEW: JSON vs TXT 自动分发
    if (ends_with(filename, ".json")) {
        std::ifstream fin(filename);
        if (!fin) throw std::runtime_error("Cannot open json: " + filename);
        json j; fin >> j;
        loadFromJsonObject(j); 
        return;
    }

    // NEW: ONNX 直读分发
    if (ends_with(filename, ".onnx")) {
        loadFromOnnxFile(filename);    
        return;
    }

    // 默认按 txt 读取（保持兼容）
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Cannot open txt: " + filename);
    loadFromTxtStream(input);

}

NeuralNetwork::NeuralNetwork(string filename, string PYTHONPATH)
{
    system((PYTHONPATH + " " + "onnx_converter" + " " + filename).c_str());
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Cannot open converted nn file: " + filename);
    loadFromTxtStream(input);
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
        // cout << "------------- Layer " << s << " starts. -------------" << endl;
        Layer layer = layers[s];
        
        TaylorModelVec<Real> tmvTemp_pre;
        TaylorModelVec<Real> tmvTemp_post;

        Variables tmvars;
        int x_id = tmvars.declareVar("x");

		 
        layer.pre_activate(tmvTemp_pre, tmv_all_layer[s], domain, polar_setting);
//        cout << "pre: " << tmvTemp_pre.tms[0].remainder << endl;

        Variables pre_vars;
        for(unsigned int i=0; i<tmvTemp_pre.tms.size(); ++i)
        {
            int var = pre_vars.declareVar("y_"+to_string(s)+"_"+to_string(i));
        }
        cout << "Pre" << endl;
        tmvTemp_pre.output(cout, pre_vars, tmvars);
    
        layer.post_activate(tmvTemp_post, tmvTemp_pre, domain, polar_setting, setting);
//        cout << "post: " << tmvTemp_post.tms[0].remainder << endl;

        Variables post_vars;
        for(unsigned int i=0; i<tmvTemp_post.tms.size(); ++i)
        {
            int var = post_vars.declareVar("y_"+to_string(s)+"_"+to_string(i));
        }
        cout << "Post" << endl;
        tmvTemp_post.output(cout, post_vars, tmvars);

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

		if (polar_setting.get_num_threads() < 0) // Single thread
		{
			for (int j=0; j<layers[k].bias.rows(); j++)
			{
				Polynomial<Real> poly_temp(layers[k].bias[j][0], domain.size());
				tmv_layer_input.tms[j].expansion += poly_temp;
			}
		} 
		else 
		{
			//cout << "Multi-Thread get_tmv_output_symbolic for layer " << k <<endl;
			//
			// Get the number of neurons
			int num_rows = layers[k].bias.rows();

			// Create placeholder for the biases and expansions
			Real bias_arr[num_rows];
			Polynomial<Real> expansions[num_rows];
			
			
			// Fillout the placeholders
			for (int j = 0; j < num_rows; j++) 
			{
				bias_arr[j] = layers[k].bias[j][0];
				expansions[j] = tmv_layer_input.tms[j].expansion;
			}
			 
			// Create multiple threads;
			//vector<thread> ths(this->num_threads);
			vector<std::thread> threads;
			if(polar_setting.get_num_threads() > 0) 
			{
				threads.reserve(polar_setting.get_num_threads());
			} 
			
			// Add biases to the expansions
			for (int j = 0; j < num_rows; j++) 
			{
				threads.emplace_back([&](int j) {
					//mtx.lock();
					Polynomial<Real> poly_temp(bias_arr[j], domain.size());
					expansions[j] += poly_temp;
				}, j);
			}
			for (auto& th: threads) 
			{
				th.join();
			}
			for (int j = 0; j < num_rows; j++)
			{
				tmv_layer_input.tms[j].expansion = expansions[j];
			}
			//cout << "-------- Multi-thread All Neurons Pre Done -------- " << endl;
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

				if (polar_setting.get_num_threads() < 0) // Single thread
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
					//cout << "Multi-Thread bern_poly coefficient computation for layer " << k <<endl;
					// Get size of input range
					unsigned int rangeDim = input_range.size();
                    Interval input_range_arr[rangeDim];
                    std::copy(input_range.begin(), input_range.end(), input_range_arr);
        
					// Create place holder for Berns polys and remainders
                    UnivariatePolynomial<Real> Berns_poly_arr[rangeDim];
					Interval Berns_rem_arr[rangeDim];
			 
					// Create multiple threads;
					//vector<thread> ths(this->num_threads);
					vector<std::thread> threads;
					if(polar_setting.get_num_threads() > 0) 
					{
						threads.reserve(polar_setting.get_num_threads());
					} 
					
					// Add biases to the expansions
					for (int j = 0; j < rangeDim; j++) 
					{
						threads.emplace_back([&](int j) {
							//UnivariatePolynomial<Real> up;
						    gen_bern_poly(Berns_poly_arr[j], layers[k].activation, input_range_arr[j], polar_setting.get_bernstein_order());
						    //Berns_poly_arr[j] = up;
						    Real error;
						    Berns_poly_arr[j].evaluate(error, Real(0));
						    Interval rem;

						    if(Berns_poly_arr[j].coefficients.size() > 2)
						    {
							    Interval I(-0.5*error, 0.5*error, 1);
							    rem = I;
							    Berns_poly_arr[j].coefficients[0] -= 0.5*error;
	                // cout << I << endl;
						    }
						    Berns_rem_arr[j] = rem;
						}, j);
					}
					for (auto& th: threads) 
					{
						th.join();
					}
					for (int j = 0; j < rangeDim; j++)
                    {
                        Berns_rem[j] = Berns_rem_arr[j];
                        Berns_poly[j] = Berns_poly_arr[j];
                    }
					//cout << "-------- Multi-thread All Neurons Pre Done -------- " << endl;
				}

			}
			else
			{
                if (polar_setting.get_num_threads() < 0) // Single thread
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
                else
                {
                    //cout << "Multi-Thread bern_poly coefficient computation for layer " << k <<endl;
					// Get size of input range
					unsigned int rangeDim = input_range.size();
                    Interval input_range_arr[rangeDim];
                    std::copy(input_range.begin(), input_range.end(), input_range_arr);
        
					// Create place holder for Berns polys and remainders
                    UnivariatePolynomial<Real> Berns_poly_arr[rangeDim];
					Interval Berns_rem_arr[rangeDim];
				 
					// Create multiple threads;
					//vector<thread> ths(this->num_threads);
					vector<std::thread> threads;
					if(polar_setting.get_num_threads() > 0) 
					{
						threads.reserve(polar_setting.get_num_threads());
					} 
                    for(int j=0; j<input_range.size(); ++j)
                    {
                        threads.emplace_back([&](int j) {
                            //UnivariatePolynomial<Real> up;
                            gen_bern_poly(Berns_poly_arr[j], layers[k].activation, input_range_arr[j], polar_setting.get_bernstein_order());

                            //Berns_poly_arr[j] = up;

                            double error = gen_bern_err_by_sample(Berns_poly_arr[j], layers[k].activation, input_range_arr[j], polar_setting.get_partition_num());

        //					cout << error << endl;

        //					if(error > 1e-5)
        //						cout << error << endl;

                            Interval rem(-error, error);
                            Berns_rem_arr[j] = rem;
                            }, j);
                    }
                    for (auto& th: threads) 
					{
						th.join();
					}
					for (int j = 0; j < rangeDim; j++)
                    {
                        Berns_rem[j] = Berns_rem_arr[j];
                        Berns_poly[j] = Berns_poly_arr[j];
                    }

                }
			}
		}





		// extracting the linear part
		Matrix<Real> Q_i(input_range.size());

		if(layers[k].activation != "Affine")
		{
            if (polar_setting.get_num_threads() < 0) // Single thread
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
            else 
            {
                int Berns_poly_size = Berns_poly.size();
                // Create place holder for Q_i diagonal
                Real Q_i_diag[Berns_poly_size];
                // Create place holder for Berns polys and remainders
                UnivariatePolynomial<Real> Berns_poly_arr[Berns_poly_size];
                std::copy(Berns_poly.begin(), Berns_poly.end(), Berns_poly_arr);
                vector<std::thread> threads;
                if(polar_setting.get_num_threads() > 0) 
                {
                    threads.reserve(polar_setting.get_num_threads());
                } 
                for(int j=0; j<Berns_poly.size(); ++j)
                {
                    threads.emplace_back([&](int j) {
                        if(Berns_poly_arr[j].coefficients.size() > 1)
                        {
                            Q_i_diag[j] = Berns_poly_arr[j].coefficients[1];
                        }
                        else
                        {
                            Q_i_diag[j] = 0;
                        }
                    }, j);
                }
                for (auto& th: threads) 
                {
                    th.join();
                }
                for (int j = 0; j < Berns_poly_size; j++)
                {
                     Q_i[j][j] = Q_i_diag[j];
                }
            }
		}


		// computing the new TM q_i
		TaylorModelVec<Real> tmvTemp;

		int order = polar_setting.get_bernstein_order();


		if(layers[k].activation != "Affine")
		{
            if (polar_setting.get_num_threads() < 0) // Single thread
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
                int Berns_poly_size = Berns_poly.size();
                // Create place holder for Berns polys, remainders and Berns Taylor Models
                UnivariatePolynomial<Real> Berns_poly_arr[Berns_poly_size];
                std::copy(Berns_poly.begin(), Berns_poly.end(), Berns_poly_arr);
                Interval Berns_rem_arr[Berns_poly_size];
                std::copy(Berns_rem.begin(), Berns_rem.end(), Berns_rem_arr);
                TaylorModel<Real> tm_berns_arr[Berns_poly_size];

                // Create place holder for input Taylor Models
                TaylorModel<Real> tm_is[Berns_poly_size];
                std::copy(tmv_layer_input.tms.begin(), tmv_layer_input.tms.end(), tm_is);

                vector<std::thread> threads;
                if(polar_setting.get_num_threads() > 0) 
                {
                    threads.reserve(polar_setting.get_num_threads());
                } 
                
                for(int j=0; j<Berns_poly_size; ++j)
                {
                    threads.emplace_back([&](int j) 
                    {
                        TaylorModel<Real> tm_berns;

                        if(Berns_poly_arr[j].coefficients.size() > 2)
                        {
                            TaylorModel<Real> tmTemp(Berns_poly_arr[j].coefficients.back(), domain.size());

                            // the nonlinear part
                            for (int i = Berns_poly_arr[j].coefficients.size() - 2; i >= 0; --i)
                            {
                                tmTemp.mul_ctrunc_assign(tm_is[j], domain, order, setting.tm_setting.cutoff_threshold);

                                if(i >= 2)
                                {
                                    TaylorModel<Real> tmTemp2(Berns_poly_arr[j].coefficients[i], domain.size());
                                    tmTemp += tmTemp2;
                                }
                            }

                            // the linear part
                            if(Berns_poly_arr[j].coefficients.size() > 1)
                            {
                                tmTemp.expansion += tm_is[j].expansion * Berns_poly_arr[j].coefficients[1];
                            }

                            // the constant
                            if(Berns_poly_arr[j].coefficients.size() > 0)
                            {
                                Polynomial<Real> polyTemp(Berns_poly_arr[j].coefficients[0], domain.size());
                                tmTemp.expansion += polyTemp;
                            }

                            tm_berns_arr[j] = tmTemp;
                        }
                        else
                        {
                            TaylorModel<Real> tmTemp;

                            // the linear part
                            if(Berns_poly_arr[j].coefficients.size() > 1)
                            {
                                tmTemp.expansion = tm_is[j].expansion * Berns_poly_arr[j].coefficients[1];
                            }

                            // the constant
                            if(Berns_poly_arr[j].coefficients.size() > 0)
                            {
                                Polynomial<Real> polyTemp(Berns_poly_arr[j].coefficients[0], domain.size());
                                tmTemp.expansion += polyTemp;
                            }

                            tm_berns_arr[j] = tmTemp;
                        }

                        tm_berns_arr[j].remainder += Berns_rem_arr[j];
                    }, j);
                }
                
                for (auto& th: threads) 
                {
                    th.join();
                }

                for(int j=0; j<Berns_poly_size; ++j)
                {
                    tmvTemp.tms.push_back(tm_berns_arr[j]);
                }
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
			if (true || polar_setting.get_num_threads() < 0) // Single thread
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
            else 
            {
                int input_size = input_range.size();
                // Create place holders for vectors
                Interval input_range_arr[input_size];
                std::copy(input_range.begin(), input_range.end(), input_range_arr);
                TaylorModel<Real> tmv_layer_input_precond_arr[input_size];
                std::copy(tmv_layer_input_precond.tms.begin(), tmv_layer_input_precond.tms.end(), tmv_layer_input_precond_arr);
                UnivariateTaylorModel<Real> utm_activation_arr[utm_activation.size()];

                vector<std::thread> threads;
                if(polar_setting.get_num_threads() > 0) 
                {
                    threads.reserve(polar_setting.get_num_threads());
                } 
                
                if(layers[k].activation == "sigmoid")
                {
                    for(int j=0; j<input_size; ++j)
                    {
                        threads.emplace_back([&](int j) 
                        {
                            Real const_part;
                            tmv_layer_input_precond_arr[j].constant(const_part);
                            tmv_layer_input_precond_arr[j].rmConstant();

                            UnivariateTaylorModel<Real> utm_x;
                            utm_x.expansion.coefficients.push_back(const_part);
                            utm_x.expansion.coefficients.push_back(1);

                            Interval x_range =  input_range_arr[j] - const_part; //input_range[j] - const_part;

                            interval_utm_setting.val = x_range;

                            utm_x.sigmoid_taylor(utm_activation_arr[j], x_range,  polar_setting.get_bernstein_order(), setting.g_setting);
                        }, j);
                    }
                }
                else if(layers[k].activation == "tanh")
                {
                    for(int j=0; j<input_size; ++j)
                    {
                        threads.emplace_back([&](int j) 
                        {
                            Real const_part;
                            tmv_layer_input_precond_arr[j].constant(const_part);
                            tmv_layer_input_precond_arr[j].rmConstant();

                            UnivariateTaylorModel<Real> utm_x;
                            utm_x.expansion.coefficients.push_back(const_part);
                            utm_x.expansion.coefficients.push_back(1);

                            Interval x_range = input_range_arr[j] - const_part;

                            interval_utm_setting.val = x_range;

                            utm_x.tanh_taylor(utm_activation_arr[j], x_range, polar_setting.get_bernstein_order(), setting.g_setting);
                        }, j);
                    }
                }
                for (auto& th: threads) 
                {
                    th.join();
                }

                for(int j=0; j<input_size; ++j)
                {
                    //tmv_layer_input_precond.tms[j].constant(tmv_layer_input_precond_arr[j].expansion.constant_part);
                    //tmv_layer_input_precond.tms[j].rmConstant();
                    utm_activation[j] = utm_activation_arr[j];
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
			if (polar_setting.get_num_threads() < 0) // Single thread
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
            else 
            {
                int utm_activation_size = utm_activation.size();
                // Create place holders for vectors
                TaylorModel<Real> tmv_layer_input_precond_arr[utm_activation_size];
                std::copy(tmv_layer_input_precond.tms.begin(), tmv_layer_input_precond.tms.end(), tmv_layer_input_precond_arr);
                UnivariateTaylorModel<Real> utm_activation_arr[utm_activation_size];
                std::copy(utm_activation.begin(), utm_activation.end(), utm_activation_arr);
                TaylorModel<Real> tm_taylors[utm_activation_size];
                vector<std::thread> threads;
                if(polar_setting.get_num_threads() > 0) 
                {
                    threads.reserve(polar_setting.get_num_threads());
                } 

                for(int j=0; j<utm_activation.size(); ++j)
                {
                    threads.emplace_back([&](int j) 
                    {
                        //TaylorModel<Real> tm_taylor;

                        if(utm_activation_arr[j].expansion.coefficients.size() > 2)
                        {
                            TaylorModel<Real> tmTemp(utm_activation_arr[j].expansion.coefficients.back(), domain.size());

                            // the nonlinear part
                            for (int i = utm_activation_arr[j].expansion.coefficients.size() - 2; i >= 0; --i)
                            {
                                tmTemp.mul_ctrunc_assign(tmv_layer_input_precond_arr[j], domain, order, setting.tm_setting.cutoff_threshold);

                                if(i >= 2)
                                {
                                    TaylorModel<Real> tmTemp2(utm_activation_arr[j].expansion.coefficients[i], domain.size());
                                    tmTemp += tmTemp2;
                                }
                            }

                            // the linear part
                            if(utm_activation_arr[j].expansion.coefficients.size() > 1)
                            {
                                tmTemp.expansion += tmv_layer_input_precond_arr[j].expansion * utm_activation_arr[j].expansion.coefficients[1];
                            }

                            // the constant
                            if(utm_activation_arr[j].expansion.coefficients.size() > 0)
                            {
                                Polynomial<Real> polyTemp(utm_activation_arr[j].expansion.coefficients[0], domain.size());
                                tmTemp.expansion += polyTemp;
                            }

                            tm_taylors[j] = tmTemp;
                        }
                        else
                        {
                            TaylorModel<Real> tmTemp;

                            // the linear part
                            if(utm_activation_arr[j].expansion.coefficients.size() > 1)
                            {
                                tmTemp.expansion = tmv_layer_input_precond_arr[j].expansion * utm_activation_arr[j].expansion.coefficients[1];
                            }

                            // the constant
                            if(utm_activation_arr[j].expansion.coefficients.size() > 0)
                            {
                                Polynomial<Real> polyTemp(utm_activation_arr[j].expansion.coefficients[0], domain.size());
                                tmTemp.expansion += polyTemp;
                            }

                            tm_taylors[j] = tmTemp;
                        }

                        tm_taylors[j].remainder += utm_activation_arr[j].remainder;
                    }, j);
                }
                for (auto& th: threads) 
                {
                    th.join();
                }

                for(int j=0; j<utm_activation_size; ++j)
                {
                    if(tm_taylors[j].remainder.width() < tmvTemp.tms[j].remainder.width())
                        {
                            tmvTemp.tms[j] = tm_taylors[j];
                            Q_i[j][j] = Q_i_Taylor[j][j];
                        }
                }
            }
		}






		Matrix<Real> Phi_i = Q_i * layers[k].weight;



		Matrix<Interval> J_k(tmvTemp.tms.size(), 1);
		tmvTemp.Remainder(J_k);


        if (polar_setting.get_num_threads() < 0) // Single thread
        {
            // updating Q
            for(int j=0; j<Q.size(); ++j)
            {
                Q[j] = Phi_i * Q[j];
            }
        }
        else 
        {
            int Q_size = Q.size();
            Matrix<Real> Q_js[Q_size];
            std::copy(Q.begin(), Q.end(), Q_js);
            vector<std::thread> threads;
            if(polar_setting.get_num_threads() > 0) 
            {
                threads.reserve(polar_setting.get_num_threads());
            } 
            for(int j=0; j<Q_size; ++j)
            {
                threads.emplace_back([&](int j) 
                {
                    Q_js[j] = Phi_i * Q_js[j];
                }, j); 
            }
            for (auto& th: threads) 
            {
                th.join();
            }
            for(int j=0; j<Q.size(); ++j)
            {
                Q[j] = Q_js[j];
            }
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



		//cout <<  "updating q_i" << endl;
		q_i = tmvTemp;

		Matrix<Interval> imTemp = Q[0] * I1;
        if (polar_setting.get_num_threads() < 0) // Single thread
        {
            for(int j=0; j<q_i.tms.size(); ++j)
            {
                q_i.tms[j].remainder += mathbb_J[j][0] + imTemp[j][0];
            }
        }else 
        {
            int q_i_size = q_i.tms.size();
            

            Interval mathbb_J_js[q_i_size], imTemp_js[q_i_size];
            for (int j=0; j < q_i_size; j++) 
            {
                mathbb_J_js[j] = mathbb_J[j][0];
                imTemp_js[j] = imTemp[j][0];
            }
            Interval remainders[q_i_size];
            vector<std::thread> threads;
            if(polar_setting.get_num_threads() > 0) 
            {
                threads.reserve(polar_setting.get_num_threads());
            } 
            for(int j=0; j<q_i_size; ++j)
            {
                threads.emplace_back([&](int j) 
                {
                    //cout << "the hell?" << endl;
                    remainders[j] = mathbb_J_js[j] + imTemp_js[j];
                }, j); 
            }
            for (auto& th: threads) 
            {
                th.join();
            }
            for (int j=0; j < q_i_size; j++) 
            {
                q_i.tms[j].remainder += remainders[j];
            }
        }

	}

	result = q_i;

    
	// cout << "applying the scalars and offset" << endl;
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

void NeuralNetwork::loadFromTxtStream(std::istream& input)
{
    std::string line;

    if (!getline(input, line)) {
        std::cout << "failed to read file: Neural Network." << std::endl;
        throw std::runtime_error("empty nn txt");
    }
    try { num_of_inputs = stoi(line); }
    catch (...) { std::cout << "string/integer conversion error!\n" << line << std::endl; throw; }

    getline(input, line); num_of_outputs = stoi(line);
    getline(input, line); num_of_hidden_layers = stoi(line);

    std::vector<int> network_structure(num_of_hidden_layers + 1, 0);
    for (int idx = 0; idx < num_of_hidden_layers; idx++) {
        getline(input, line);
        network_structure[idx] = stoi(line);
    }
    network_structure.back() = num_of_outputs;

    std::vector<std::string> activation;
    activation.reserve(num_of_hidden_layers + 1);
    for (int idx = 0; idx < num_of_hidden_layers + 1; idx++) {
        getline(input, line);
        activation.push_back(line);
    }

    // Input layer
    Matrix<Real> weight0(network_structure[0], num_of_inputs);
    Matrix<Real> bias0(network_structure[0], 1);
    for (int i = 0; i < network_structure[0]; i++) {
        for (int j = 0; j < num_of_inputs; j++) { getline(input, line); weight0[i][j] = stod(line); }
        getline(input, line); bias0[i][0] = stod(line);
    }
    layers.clear();
    layers.emplace_back(activation[0], weight0, bias0);

    // Hidden + output layer
    for (int layer_idx = 0; layer_idx < num_of_hidden_layers; layer_idx++) {
        Matrix<Real> weight(network_structure[layer_idx + 1], network_structure[layer_idx]);
        Matrix<Real> bias(network_structure[layer_idx + 1], 1);
        for (int i = 0; i < network_structure[layer_idx + 1]; i++) {
            for (int j = 0; j < network_structure[layer_idx]; j++) { getline(input, line); weight[i][j] = stod(line); }
            getline(input, line); bias[i][0] = stod(line);
        }
        layers.emplace_back(activation[layer_idx + 1], weight, bias);
    }

    getline(input, line); offset = stod(line);
    getline(input, line); scale_factor = stod(line);
}


// NEW: Import from Json object
void NeuralNetwork::loadFromJsonObject(const json& j)
{
    // 基本字段
    if (!j.contains("type") || tolower_str(j.at("type").get<std::string>()) != "mlp")
        throw std::runtime_error("JSON NN must have type='mlp'");
    num_of_inputs  = j.at("in_dim").get<int>();
    num_of_outputs = j.at("out_dim").get<int>();
    offset         = j.value("offset", 0.0);
    scale_factor   = j.value("scale", 1.0);

    const auto& JL = j.at("layers");
    if (!JL.is_array() || JL.empty()) throw std::runtime_error("JSON layers empty");

    layers.clear();
    int prev_in = num_of_inputs;
    int hidden_count = 0;

    for (size_t li = 0; li < JL.size(); ++li) {
        const auto& L = JL[li];
        if (!L.contains("W") || !L.contains("b"))
            throw std::runtime_error("JSON layer missing W/b");

        // Read weights
        const auto& Wj = L.at("W");
        if (!Wj.is_array() || Wj.empty()) throw std::runtime_error("W empty");
        int out_dim = (int)Wj.size();
        int in_dim  = (int)Wj.at(0).size();
        if (in_dim != prev_in)
            throw std::runtime_error("in_dim mismatch at layer " + std::to_string(li));

        Matrix<Real> W(out_dim, in_dim);
        for (int r = 0; r < out_dim; ++r) {
            if ((int)Wj.at(r).size() != in_dim) throw std::runtime_error("W row size mismatch");
            for (int c = 0; c < in_dim; ++c) {
                W[r][c] = Wj.at(r).at(c).get<double>();
            }
        }

        // Read bias
        const auto& bj = L.at("b");
        if ((int)bj.size() != out_dim) throw std::runtime_error("b size mismatch");
        Matrix<Real> B(out_dim, 1);
        for (int r = 0; r < out_dim; ++r) B[r][0] = bj.at(r).get<double>();

        // Read activation
        std::string act = map_activation_json_to_internal(L.value("act", "linear"));

        // Into stack
        layers.emplace_back(act, W, B);
        prev_in = out_dim;

        // Number of hidden layer
        if ((int)li < (int)JL.size() - 1) hidden_count++;
    }

    num_of_hidden_layers = hidden_count; 
}


void NeuralNetwork::loadFromOnnxFile(const std::string& path)
{
#ifndef USE_ONNX_PROTO
    throw std::runtime_error("ONNX direct parse not enabled. Build with -DUSE_ONNX_PROTO.");
#else
    offset = 0.0;
    scale_factor = 1.0;
    layers.clear();

    // 1) 读取 ONNX protobuf
    onnx::ModelProto model;
    {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) throw std::runtime_error("Cannot open ONNX: " + path);
        google::protobuf::io::IstreamInputStream is(&ifs);
        if (!model.ParseFromZeroCopyStream(&is)) {
            throw std::runtime_error("Failed to parse ONNX protobuf: " + path);
        }
    }
    const onnx::GraphProto& G = model.graph();

    // 2) 收集 initializers（name -> tensor）
    std::unordered_map<std::string, const onnx::TensorProto*> init;
    init.reserve((size_t)G.initializer_size());
    for (const auto& t : G.initializer()) init.emplace(t.name(), &t);

    // 输出端仿射累积： y <- out_a * y + out_b
    double out_a = 1.0, out_b = 0.0;

    // 读取 initializer 的“标量或全等一维向量” -> double
    auto read_uniform_const = [&](const std::string& name, double& val)->bool {
        auto it = init.find(name);
        if (it == init.end()) return false;
        const onnx::TensorProto* tp = it->second;

        std::vector<double> flat;
        auto load_float = [&](const onnx::TensorProto* t){
            if (t->has_raw_data()) {
                const std::string& raw = t->raw_data();
                size_t n = raw.size() / sizeof(float);
                std::vector<float> buf(n);
                std::memcpy(buf.data(), raw.data(), n*sizeof(float));
                flat.resize(n);
                for (size_t i=0;i<n;++i) flat[i] = static_cast<double>(buf[i]);
            } else {
                const int n = t->float_data_size();
                flat.resize(n);
                for (int i=0;i<n;++i) flat[i] = static_cast<double>(t->float_data(i));
            }
        };
        auto load_double = [&](const onnx::TensorProto* t){
            if (t->has_raw_data()) {
                const std::string& raw = t->raw_data();
                size_t n = raw.size() / sizeof(double);
                flat.resize(n);
                std::memcpy(flat.data(), raw.data(), n*sizeof(double));
            } else {
                const int n = t->double_data_size();
                flat.resize(n);
                for (int i=0;i<n;++i) flat[i] = t->double_data(i);
            }
        };
        if (tp->data_type() == onnx::TensorProto::FLOAT) load_float(tp);
        else if (tp->data_type() == onnx::TensorProto::DOUBLE) load_double(tp);
        else return false;

        if (flat.empty()) return false;
        double v0 = flat[0];
        for (double v : flat) if (std::fabs(v - v0) > 0.0) return false; // 必须全等
        val = v0;
        return true;
    };


    // ---- 辅助：读取 tensor 为 Matrix<Real>（2D）----
    auto tensor_to_matrix = [](const onnx::TensorProto* tp, int64_t& rows, int64_t& cols) -> Matrix<Real> {
        if (!tp) throw std::runtime_error("Null tensor proto");
        std::vector<double> flat;

        auto load_float = [&](const onnx::TensorProto* t){
            if (t->has_raw_data()) {
                const std::string& raw = t->raw_data();
                size_t n = raw.size() / sizeof(float);
                std::vector<float> buf(n);
                std::memcpy(buf.data(), raw.data(), n*sizeof(float));
                flat.resize(n);
                for (size_t i=0;i<n;++i) flat[i] = static_cast<double>(buf[i]);
            } else {
                const int n = t->float_data_size();
                flat.resize(n);
                for (int i=0;i<n;++i) flat[i] = static_cast<double>(t->float_data(i));
            }
        };
        auto load_double = [&](const onnx::TensorProto* t){
            if (t->has_raw_data()) {
                const std::string& raw = t->raw_data();
                size_t n = raw.size() / sizeof(double);
                flat.resize(n);
                std::memcpy(flat.data(), raw.data(), n*sizeof(double));
            } else {
                const int n = t->double_data_size();
                flat.resize(n);
                for (int i=0;i<n;++i) flat[i] = t->double_data(i);
            }
        };

        if (tp->data_type() == onnx::TensorProto::FLOAT)      load_float(tp);
        else if (tp->data_type() == onnx::TensorProto::DOUBLE) load_double(tp);
        else throw std::runtime_error("Unsupported tensor dtype (expect FLOAT/DOUBLE).");

        if (tp->dims_size() != 2) throw std::runtime_error("Expect 2D weight tensor.");
        rows = tp->dims(0);
        cols = tp->dims(1);
        if ((int64_t)flat.size() != rows*cols) throw std::runtime_error("Tensor size mismatch.");

        Matrix<Real> M((unsigned)rows, (unsigned)cols);
        size_t k=0;
        for (unsigned r=0;r<(unsigned)rows;++r)
            for (unsigned c=0;c<(unsigned)cols;++c)
                M[r][c] = flat[k++];
        return M;
    };

    // ---- 辅助：读取 bias 为 [out,1] 的 Matrix<Real> ----
    auto tensor_to_bias_col = [](const onnx::TensorProto* tp, int64_t& out_dim) -> Matrix<Real> {
        if (!tp) throw std::runtime_error("Null bias tensor");
        std::vector<double> flat;

        auto load_float = [&](const onnx::TensorProto* t){
            if (t->has_raw_data()) {
                const std::string& raw = t->raw_data();
                size_t n = raw.size() / sizeof(float);
                std::vector<float> buf(n);
                std::memcpy(buf.data(), raw.data(), n*sizeof(float));
                flat.resize(n);
                for (size_t i=0;i<n;++i) flat[i] = static_cast<double>(buf[i]);
            } else {
                const int n = t->float_data_size();
                flat.resize(n);
                for (int i=0;i<n;++i) flat[i] = static_cast<double>(t->float_data(i));
            }
        };
        auto load_double = [&](const onnx::TensorProto* t){
            if (t->has_raw_data()) {
                const std::string& raw = t->raw_data();
                size_t n = raw.size() / sizeof(double);
                flat.resize(n);
                std::memcpy(flat.data(), raw.data(), n*sizeof(double));
            } else {
                const int n = t->double_data_size();
                flat.resize(n);
                for (int i=0;i<n;++i) flat[i] = t->double_data(i);
            }
        };

        if (tp->data_type() == onnx::TensorProto::FLOAT)      load_float(tp);
        else if (tp->data_type() == onnx::TensorProto::DOUBLE) load_double(tp);
        else throw std::runtime_error("Unsupported bias dtype.");

        if (tp->dims_size()==1) out_dim = tp->dims(0);
        else if (tp->dims_size()==2) out_dim = std::max<int64_t>(tp->dims(0), tp->dims(1));
        else throw std::runtime_error("Bias dims unsupported.");

        if ((int64_t)flat.size() != out_dim) throw std::runtime_error("Bias size mismatch.");

        Matrix<Real> B((unsigned)out_dim, 1u);
        for (unsigned r=0;r<(unsigned)out_dim;++r) B[r][0] = flat[r];
        return B;
    };

    auto map_act_from_onnx = [](const std::string& op)->std::string{
        std::string a = tolower_str(op);
        if (a=="relu") return "ReLU";
        if (a=="tanh") return "tanh";
        if (a=="sigmoid") return "sigmoid";
        if (a=="identity" || a=="linear") return "Affine";
        return "Affine";
    };

    // 收集“第一层之前”的 x 预处理：x' = pre_s * x + pre_o   （仅支持标量，足够覆盖我们导出的 ONNX）
    double pre_s = 1.0;
    double pre_o = 0.0;
    Real pre_sR = Real(1.0);
    Real pre_oR = Real(0.0);
    bool   first_linear_seen = false;

    // 小工具：从 initializer 读一个标量 double（支持形如 [], [1]）
    auto read_scalar_from_init = [&](const std::string& name, double& out)->bool {
        auto it = init.find(name);
        if (it == init.end()) return false;
        const onnx::TensorProto* tp = it->second;
        // 允许标量、长度为1的一维或二维
        size_t n_elems = 0;
        if (tp->has_raw_data()) {
            const std::string& raw = tp->raw_data();
            if (tp->data_type() == onnx::TensorProto::FLOAT) {
                if (raw.size() != sizeof(float)) return false;
                float v; std::memcpy(&v, raw.data(), sizeof(float));
                out = static_cast<double>(v); return true;
            } else if (tp->data_type() == onnx::TensorProto::DOUBLE) {
                if (raw.size() != sizeof(double)) return false;
                double v; std::memcpy(&v, raw.data(), sizeof(double));
                out = v; return true;
            } else return false;
        } else {
            if (tp->data_type() == onnx::TensorProto::FLOAT) {
                if (tp->float_data_size() != 1) return false;
                out = static_cast<double>(tp->float_data(0)); return true;
            } else if (tp->data_type() == onnx::TensorProto::DOUBLE) {
                if (tp->double_data_size() != 1) return false;
                out = tp->double_data(0); return true;
            } else return false;
        }
    };

    // 3) 解析节点：Gemm 或 MatMul(+Add) + 激活
    bool pending_activation = false;
    int  in_dim = -1, out_dim_decl = -1;

    for (int ni = 0; ni < G.node_size(); ++ni) {
        const onnx::NodeProto& node = G.node(ni);
        std::string op = tolower_str(node.op_type());

        // 0) 第一层之前：输入端标量 Mul/Add 折叠（可有可无），以及纯形状算子忽略
        if (!first_linear_seen) {
            if (op == "mul" || op == "add") {
                double cst; bool has_scalar = false;
                if (node.input_size() == 2) {
                    has_scalar = read_scalar_from_init(node.input(0), cst) || read_scalar_from_init(node.input(1), cst);
                }
                if (has_scalar) {
                    if (op == "mul") { pre_s *= cst; pre_o *= cst; pre_sR *= Real(cst); pre_oR *= Real(cst); }
                    else            { pre_o += cst; pre_oR += Real(cst); }
                    continue;
                }
            }
            if (op == "unsqueeze" || op == "squeeze" || op == "reshape" || op == "flatten" || op == "identity") {
                continue; // 纯形状对数值无影响
            }
        }

        // 1) 线性层：Gemm / MatMul  —— 必须入栈并立刻 continue
        if (op == "gemm" || op == "matmul") {
            if (node.input_size() < 2)
                throw std::runtime_error("Linear op needs >=2 inputs.");

            const std::string& Wname = node.input(1);
            auto itW = init.find(Wname);
            if (itW == init.end()) throw std::runtime_error("Cannot find weight initializer: " + Wname);
            const onnx::TensorProto* Wtp = itW->second;

            bool   transB = false;
            double alpha  = 1.0, beta = 1.0;
            if (op == "gemm") {
                for (const auto& a : node.attribute()) {
                    std::string an = tolower_str(a.name());
                    if (an=="transb") transB = (a.i()!=0);
                    else if (an=="alpha") alpha = a.f();
                    else if (an=="beta")  beta  = a.f();
                }
            }

            // 读权重
            int64_t Wr, Wc;
            Matrix<Real> W = tensor_to_matrix(Wtp, Wr, Wc);

            // 统一内部为 [out, in]
            if (op == "gemm") {
                if (transB == 0) { Matrix<Real> WT; W.transpose(WT); W = WT; std::swap(Wr, Wc); }
                // transB==1: W 已是 [out,in]，不转置
            } else { // matmul
                // 常见导出 W 为 [in,out]，转置成 [out,in]
                Matrix<Real> WT; W.transpose(WT); W = WT; std::swap(Wr, Wc);
            }

            // alpha
            if (alpha != 1.0) {
                const Real alphaR = Real(alpha);
                for (unsigned r=0; r<(unsigned)Wr; ++r)
                    for (unsigned c=0; c<(unsigned)Wc; ++c)
                        W[r][c] *= alphaR;
            }

            // 初始化 bias，全 0
            Matrix<Real> B((unsigned)Wr, 1u);
            for (unsigned r=0; r<(unsigned)Wr; ++r) B[r][0] = Real(0.0);

            // Gemm 的第三输入 bias（若有）
            if (op == "gemm" && node.input_size() >= 3) {
                const std::string& bname = node.input(2);
                auto itb = init.find(bname);
                if (itb != init.end()) {
                    int64_t bdim;
                    Matrix<Real> Bb = tensor_to_bias_col(itb->second, bdim);
                    if (beta != 1.0) {
                        const Real betaR = Real(beta);
                        for (unsigned r=0; r<(unsigned)bdim; ++r) Bb[r][0] *= betaR;
                    }
                    if ((int)bdim != (int)Wr) throw std::runtime_error("Bias dim mismatch.");
                    for (unsigned r=0; r<(unsigned)Wr; ++r) B[r][0] += Bb[r][0];
                }
            }

            // 第一层：可选择折叠输入端仿射（如果 pre_s/pre_o ≠ 1/0）
            if (!first_linear_seen) {
                if (pre_s != 1.0) {
                    for (unsigned r=0; r<(unsigned)Wr; ++r)
                        for (unsigned c=0; c<(unsigned)Wc; ++c)
                            W[r][c] *= pre_sR;
                }
                if (pre_o != 0.0) {
                    for (unsigned r=0; r<(unsigned)Wr; ++r) {
                        Real rowsum = Real(0.0);
                        for (unsigned c=0; c<(unsigned)Wc; ++c) rowsum += W[r][c];
                        B[r][0] += rowsum * pre_oR;
                    }
                }
                first_linear_seen = true;
            }

            // 入栈并标记“等待激活或层内 bias 的 Add”
            layers.emplace_back("Affine", W, B);
            pending_activation = true;
            out_dim_decl = (int)Wr;
            continue;  // ←← 很重要：线性层处理完立刻跳下一个节点
        }

        // 2) 层内 bias：MatMul 后紧跟的 Add 常量（仅在 pending_activation==true 时合并）
        if (op == "add" && pending_activation) {
            if (node.input_size()==2) {
                const std::string& a0 = node.input(0);
                const std::string& a1 = node.input(1);
                const onnx::TensorProto* t0 = init.count(a0)?init[a0]:nullptr;
                const onnx::TensorProto* t1 = init.count(a1)?init[a1]:nullptr;
                const onnx::TensorProto* btp = t0 ? t0 : t1;
                if (btp) {
                    int64_t bdim;
                    Matrix<Real> Bb = tensor_to_bias_col(btp, bdim);
                    Matrix<Real>& Bold = layers.back().bias;
                    if (Bold.rows() != (unsigned)bdim || Bold.cols()!=1u)
                        throw std::runtime_error("Bias add dim mismatch.");
                    for (unsigned r=0;r<(unsigned)bdim;++r) Bold[r][0] += Bb[r][0];
                    continue; // 合并完就跳过
                }
            }
            // 如果不是常量 bias，不在这里处理，留给后面的输出仿射/忽略
        }

        // 3) 激活：只在 pending_activation==true 时接受
        if ((op=="relu" || op=="tanh" || op=="sigmoid" || op=="identity") && pending_activation) {
            layers.back().activation = map_act_from_onnx(node.op_type());
            pending_activation = false;
            continue;
        }

        // 4) 输出端仿射：在 first_linear_seen 之后吸收
        // Mul/Sub：无论 pending_activation 是 true/false 都吸收；
        // Add：只在 !pending_activation 时吸收（避免吞掉层内 bias）
        if (first_linear_seen && (op=="mul" || op=="sub" || (op=="add" && !pending_activation))) {
            if (node.input_size() == 2) {
                double c; bool ok = false;
                ok = read_uniform_const(node.input(0), c) || read_uniform_const(node.input(1), c);
                if (ok) {
                    if (op=="mul") { out_a *= c; out_b *= c; continue; }
                    else if (op=="sub") {
                        // 仅当常量在第二个输入时视为 y - c
                        if (read_uniform_const(node.input(1), c)) { out_b -= c; continue; }
                    } else if (op=="add") { out_b += c; continue; }
                }
            }
        }

        // 5) 其他（形状/Dropout 等）忽略
        // ...
    }

    for (size_t i = 1; i < layers.size(); ++i) {
        unsigned in_cols  = layers[i].weight.cols();
        unsigned prev_out = layers[i-1].weight.rows();
        if (in_cols != prev_out) {
            std::ostringstream oss;
            oss << "Layer dim mismatch at layer " << i
                << ": W.cols=" << in_cols
                << " but previous W.rows=" << prev_out;
            throw std::runtime_error(oss.str());
        }
    }

    if (layers.empty())
        throw std::runtime_error("No layers parsed from ONNX (expect an MLP).");

    // —— 将 y <- out_a * y + out_b 转换为 POLAR 的约定：scale*(y - offset) ——
    // out_a * (y - (-out_b/out_a)) = out_a*y + out_b
    scale_factor = out_a;
    if (std::fabs(out_a) > 0.0) offset = -out_b / out_a;
    else offset = 0.0; // 退化情况

    
    num_of_inputs       = (in_dim>0) ? in_dim : (int)layers.front().weight.cols();
    num_of_outputs      = (out_dim_decl>0) ? out_dim_decl : (int)layers.back().weight.rows();
    num_of_hidden_layers= std::max(0, (int)layers.size()-1);
#endif
}
