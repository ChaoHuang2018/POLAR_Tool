#include "Neuron.h"

using namespace flowstar;
using namespace std;


Neuron::Neuron() {
    
}

Neuron::Neuron(string act) {
    activation = act;
}

void Neuron::taylor_model_approx(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting, string verbose) const {
    
	if (activation == "sigmoid")
	{
		this->sigmoid_taylor(result, input, domain, polar_setting, setting, verbose);
	}
	else if (activation == "tanh")
	{
		this->tanh_taylor(result, input, domain, polar_setting, setting, verbose);
	}
	else if (activation == "ReLU")
	{
		this->relu_taylor(result, input, domain, polar_setting, setting, verbose);
	}
	else if (activation == "Affine")
	{
		this->affine_taylor(result, input, domain, polar_setting, setting, verbose);
	}
	else
	{
		cout << "The activation fundtion can be parsed." << endl;
	}
}

void Neuron::sigmoid_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting, string verbose) const
{
    unsigned int taylor_order = polar_setting.get_taylor_order();
    unsigned int bernstein_order = polar_setting.get_bernstein_order();
    unsigned int partition_num = polar_setting.get_partition_num();
    unsigned int neuron_approx_type = polar_setting.get_neuron_approx_type();
    unsigned int remainder_type = polar_setting.get_remainder_type();
    
    Interval tmRange;
    input.intEval(tmRange, domain);

	
    UnivariatePolynomial<Real> up;
    
    gen_bern_poly(up, "sigmoid", tmRange, bernstein_order);

	double error = gen_bern_err_by_sample(up, "sigmoid", tmRange, partition_num);

	Interval rem(-error, error);

//	cout << "up: " << up << endl;
	TaylorModel<Real> tmTemp(up.coefficients[up.coefficients.size() - 1], domain.size());

    // cout << "cut: " << setting.tm_setting.cutoff_threshold << endl;
	for (int i = up.coefficients.size() - 2; i >= 0; --i)
	{
		tmTemp.mul_ctrunc_assign(input, domain, taylor_order, setting.tm_setting.cutoff_threshold);
//        cout << "input: " << i << ", " << input.remainder << endl;
//        cout << "tmTemp: " << i << ", " << tmTemp.remainder << endl;

		TaylorModel<Real> tmTemp2(up.coefficients[i], domain.size());
		tmTemp += tmTemp2;
	}
    
	TaylorModel<Real> result_berns;
	result_berns = tmTemp;
	result_berns.remainder += rem;
//    cout << "rem: " << rem << endl;
    

	// cout << "Berns time: " << seconds << " seconds" << endl;
  
//    Variables vars;
//    vars.declareVar("t");
//    int x0_id = vars.declareVar("x0");
//    int x1_id = vars.declareVar("x1");
//    int x2_id = vars.declareVar("x2");
//    int x3_id = vars.declareVar("x3");
//    int x4_id = vars.declareVar("x4");
//    int x5_id = vars.declareVar("x5");
//    int u0_id = vars.declareVar("u0");
//    int u1_id = vars.declareVar("u1");
//    int u2_id = vars.declareVar("u2");
    
	TaylorModel<Real> tmTemp1 = (input) * (-1);
//    tmTemp1.output(cout, vars);
//    cout << endl;
	TaylorModel<Real> tmTemp2;
//    cout << "domain: " << domain[0] << ", " << domain[1] << endl;
//    cout << "taylor_order: " << taylor_order << endl;
//    cout << "setting.tm_setting.cutoff_threshold: " << setting.tm_setting.cutoff_threshold << endl;
	tmTemp1.exp_taylor(tmTemp2, domain, taylor_order, setting.tm_setting.cutoff_threshold, setting.g_setting);
//    tmTemp2.output(cout, vars);
    
//    cout << "tmTemp2: " << tmTemp2.remainder << endl;

	// cout << "tmTemp2: " << tmTemp2.expansion.terms.size() << endl;
	tmTemp2 += 1;
	// cout << "tmTemp2+1: " << tmTemp2.expansion.terms.size() << endl;

	TaylorModel<Real> result_taylor;
	tmTemp2.rec_taylor(result_taylor, domain, taylor_order, setting.tm_setting.cutoff_threshold, setting.g_setting);
//    result_taylor.output(cout, vars);

//    cout << "result_taylor: " << result_taylor.remainder << endl;
    // exit(0);
	// cout << "Taylor time: " << seconds << " seconds" << endl;

	if (neuron_approx_type == 1)//"Berns")
	{
		result = result_berns;
        if (verbose == "on")
        {
            cout << "BP poly: " << up << ", BP remainder:" << rem << endl;
            cout << "TM remainder after compose by BP: " << result_berns.remainder << endl;
        }
	}
	else if (neuron_approx_type == 2)//"Taylor")
	{
		result = result_taylor;
        if (verbose == "on")
        {
        }
	}
	else
	{
		if (result_berns.remainder.width() < result_taylor.remainder.width())
		{
			result = result_berns;
			// cout << "Berns" << endl;
		}
		else
		{
			result = result_taylor;
			// cout << "Taylor" << endl;
		}
	}
//    cout << "after activation, remainder: " << result.remainder << endl;
}

void Neuron::tanh_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting, string verbose) const
{
    unsigned int taylor_order = polar_setting.get_taylor_order();
    unsigned int bernstein_order = polar_setting.get_bernstein_order();
    unsigned int partition_num = polar_setting.get_partition_num();
    unsigned int neuron_approx_type = polar_setting.get_neuron_approx_type();
    unsigned int remainder_type = polar_setting.get_remainder_type();
    
    Interval tmRange;
    input.intEval(tmRange, domain);
    
//    cout << "tmRange: " << tmRange << endl;
    UnivariatePolynomial<Real> up;
    gen_bern_poly(up, "tanh", tmRange, bernstein_order);
    
//    cout << "Bernstein Polynomial: " << up << endl;

    double error = gen_bern_err_by_sample(up, "tanh", tmRange, partition_num);
//    cout << error << endl;

    Interval rem(-error, error);
//    cout << "up.coefficients.size(): " << up.coefficients.size() << endl;

    TaylorModel<Real> tmTemp(up.coefficients[up.coefficients.size() - 1], domain.size());
    
//    cout << "Bernstein Polynomial: " << up << endl;
//    cout << "111" << endl;
//    cout << "up.coefficients.size(): " << up.coefficients.size() << endl;
    for (int i = up.coefficients.size() - 2; i >= 0; --i)
    {
        tmTemp.mul_ctrunc_assign(input, domain, taylor_order, setting.tm_setting.cutoff_threshold);

        TaylorModel<Real> tmTemp2(up.coefficients[i], domain.size());
        tmTemp += tmTemp2;
    }
//    cout << "222" << endl;

    TaylorModel<Real> result_berns;
    result_berns = tmTemp;
    result_berns.remainder += rem;
    
    

    TaylorModel<Real> result_taylor;
    TaylorModel<Real> tmTemp1 = (input) * (2);
    TaylorModel<Real> tmTemp2;
    tmTemp1.exp_taylor(tmTemp2, domain, taylor_order, setting.tm_setting.cutoff_threshold, setting.g_setting);
//    cout << "11111111" << endl;
    if (tmTemp2.expansion.terms.size() == 0)
    {
        Polynomial<Real> tmp_poly(1, domain.size());
        tmTemp2.expansion = tmp_poly;
    } else
    {
        tmTemp2 += 1;
    }
    // tmTemp2 += 1;

    TaylorModel<Real> tmTemp3;
    tmTemp2.rec_taylor(tmTemp3, domain, taylor_order, setting.tm_setting.cutoff_threshold, setting.g_setting);
    TaylorModel<Real> tmTemp4 = tmTemp3 * (-2);
    if (tmTemp4.expansion.terms.size() == 0)
    {
        Polynomial<Real> tmp_poly(1, domain.size());
        tmTemp4.expansion = tmp_poly;
    } else
    {
        tmTemp4 += 1;
    }
//    cout << "33333" << endl;

    result_taylor = tmTemp4;

    if (neuron_approx_type == 1)//"Berns")
    {
        result = result_berns;
    }
    else if (neuron_approx_type == 2)//"Taylor")
    {
        result = result_taylor;
    }
    else
    {
        if (result_berns.remainder.width() < result_taylor.remainder.width())
        {
            result = result_berns;
            // cout << "Berns" << endl;
        }
        else
        {
            result = result_taylor;
            // cout << "Taylor" << endl;
        }
    }
}

void Neuron::relu_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting, string verbose) const
{
    unsigned int taylor_order = polar_setting.get_taylor_order();
    unsigned int bernstein_order = polar_setting.get_bernstein_order();
    unsigned int partition_num = polar_setting.get_partition_num();
    unsigned int neuron_approx_type = polar_setting.get_neuron_approx_type();
    unsigned int remainder_type = polar_setting.get_remainder_type();
    
    Interval tmRange;
    input.intEval(tmRange, domain);
    
    UnivariatePolynomial<Real> up;
    gen_bern_poly(up, "ReLU", tmRange, bernstein_order);

    double error = gen_bern_err_by_sample(up, "ReLU", tmRange, partition_num);

    Interval rem(-0.5*error, 0.5*error);
    up.coefficients[0] -= 0.5*error;

    TaylorModel<Real> tmTemp(up.coefficients[up.coefficients.size() - 1], domain.size());

    for (int i = up.coefficients.size() - 2; i >= 0; --i)
    {
        tmTemp.mul_ctrunc_assign(input, domain, taylor_order, setting.tm_setting.cutoff_threshold);

        TaylorModel<Real> tmTemp2(up.coefficients[i], domain.size());
        tmTemp += tmTemp2;
    }
    
    
    result = tmTemp;
    // cout << "Coeff length: " << result.expansion.terms.size() << endl;
    result.remainder += rem;
//    cout << "after activation, remainder: " << result.remainder << endl;
}

void Neuron::affine_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting, string verbose) const
{
    result = input;
    
//    unsigned int taylor_order = polar_setting.get_taylor_order();
//    unsigned int bernstein_order = polar_setting.get_bernstein_order();
//    unsigned int partition_num = polar_setting.get_partition_num();
//    string neuron_approx_type = polar_setting.get_neuron_approx_type();
//    string remainder_type = polar_setting.get_remainder_type();
//
//    vector<Real> coe;
//    coe.push_back(Real(0.0));
//    coe.push_back(Real(1.0));
//    UnivariatePolynomial<Real> up(coe);
//
//    Interval rem(0);
//
//    TaylorModel<Real> tmTemp(up.coefficients[up.coefficients.size() - 1], domain.size());
//
//    for (int i = up.coefficients.size() - 2; i >= 0; --i)
//    {
//
//        tmTemp.mul_ctrunc_assign(input, domain, taylor_order, setting.tm_setting.cutoff_threshold);
//
//        TaylorModel<Real> tmTemp2(up.coefficients[i], domain.size());
//        tmTemp += tmTemp2;
//    }
//    result = tmTemp;
//    // cout << "Coeff length: " << result.expansion.terms.size() << endl;
//    result.remainder += rem;
}

