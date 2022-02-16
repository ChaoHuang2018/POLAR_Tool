#include "Neuron.h"

using namespace flowstar;
using namespace std;


Neuron::Neuron() {
    
}

Neuron::Neuron(string act) {
    activation = act;
}

void Neuron::taylor_model_approx(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const {
    
	if (activation == "sigmoid")
	{
		this->sigmoid_taylor(result, input, domain, polar_setting, setting);
	}
	else if (activation == "tanh")
	{
		this->tanh_taylor(result, input, domain, polar_setting, setting);
	}
	else if (activation == "ReLU")
	{
		this->relu_taylor(result, input, domain, polar_setting, setting);
	}
	else if (activation == "Affine")
	{
		this->affine_taylor(result, input, domain, polar_setting, setting);
	}
	else
	{
		cout << "The activation fundtion can be parsed." << endl;
	}
}

void Neuron::sigmoid_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const
{
    unsigned int taylor_order = polar_setting.get_taylor_order();
    unsigned int bernstein_order = polar_setting.get_bernstein_order();
    unsigned int partition_num = polar_setting.get_partition_num();
    string neuron_approx_type = polar_setting.get_neuron_approx_type();
    string remainder_type = polar_setting.get_remainder_type();
    
    Interval tmRange;
    input.intEval(tmRange, domain);

	
	UnivariatePolynomial<Real> up = gen_bern_poly("sigmoid", tmRange, bernstein_order);

	double error = gen_bern_err_by_sample(up, "sigmoid", tmRange, partition_num);

	Interval rem(-error, error);

	// cout << "up: " << up << endl;
	TaylorModel<Real> tmTemp(up.coefficients[up.coefficients.size() - 1], domain.size());

	for (int i = up.coefficients.size() - 2; i >= 0; --i)
	{
		tmTemp.mul_ctrunc_assign(input, domain, taylor_order, setting.tm_setting.cutoff_threshold);

		TaylorModel<Real> tmTemp2(up.coefficients[i], domain.size());
		tmTemp += tmTemp2;
	}

	TaylorModel<Real> result_berns;
	result_berns = tmTemp;
	result_berns.remainder += rem;

	// cout << "Berns time: " << seconds << " seconds" << endl;

	TaylorModel<Real> tmTemp1 = (input) * (-1);
	TaylorModel<Real> tmTemp2;
	tmTemp1.exp_taylor(tmTemp2, domain, taylor_order, setting.tm_setting.cutoff_threshold, setting.g_setting);

	// cout << "tmTemp2: " << tmTemp2.expansion.terms.size() << endl;
	tmTemp2 += 1;
	// cout << "tmTemp2+1: " << tmTemp2.expansion.terms.size() << endl;

	TaylorModel<Real> result_taylor;
	tmTemp2.rec_taylor(result_taylor, domain, taylor_order, setting.tm_setting.cutoff_threshold, setting.g_setting);

	// cout << "Taylor time: " << seconds << " seconds" << endl;

	if (neuron_approx_type == "Berns")
	{
		result = result_berns;
	}
	else if (neuron_approx_type == "Taylor")
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

void Neuron::tanh_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const
{
    unsigned int taylor_order = polar_setting.get_taylor_order();
    unsigned int bernstein_order = polar_setting.get_bernstein_order();
    unsigned int partition_num = polar_setting.get_partition_num();
    string neuron_approx_type = polar_setting.get_neuron_approx_type();
    string remainder_type = polar_setting.get_remainder_type();
    
    Interval tmRange;
    input.intEval(tmRange, domain);
    
    UnivariatePolynomial<Real> up = gen_bern_poly("tanh", tmRange, bernstein_order);

    double error = gen_bern_err_by_sample(up, "tanh", tmRange, partition_num);

    Interval rem(-error, error);

    TaylorModel<Real> tmTemp(up.coefficients[up.coefficients.size() - 1], domain.size());

    for (int i = up.coefficients.size() - 2; i >= 0; --i)
    {
        tmTemp.mul_ctrunc_assign(input, domain, taylor_order, setting.tm_setting.cutoff_threshold);

        TaylorModel<Real> tmTemp2(up.coefficients[i], domain.size());
        tmTemp += tmTemp2;
    }

    TaylorModel<Real> result_berns;
    result_berns = tmTemp;
    result_berns.remainder += rem;

    TaylorModel<Real> result_taylor;
    TaylorModel<Real> tmTemp1 = (input) * (2);
    TaylorModel<Real> tmTemp2;
    tmTemp1.exp_taylor(tmTemp2, domain, taylor_order, setting.tm_setting.cutoff_threshold, setting.g_setting);
    // cout << "11111111" << endl;
    if (tmTemp2.expansion.terms.size() == 0)
    {
        tmTemp2.expansion.terms.push_back(Real(0));
        //cout << tmTemp2.expansion.terms.size() << endl;
        //cout << tmTemp2.remainder << endl;
    }
    tmTemp2 += 1;

    TaylorModel<Real> tmTemp3;
    tmTemp2.rec_taylor(tmTemp3, domain, taylor_order, setting.tm_setting.cutoff_threshold, setting.g_setting);
    TaylorModel<Real> tmTemp4 = tmTemp3 * (-2);
    tmTemp4 += 1;

    result_taylor = tmTemp4;

    if (neuron_approx_type == "Berns")
    {
        result = result_berns;
    }
    else if (neuron_approx_type == "Taylor")
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

void Neuron::relu_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const
{
    unsigned int taylor_order = polar_setting.get_taylor_order();
    unsigned int bernstein_order = polar_setting.get_bernstein_order();
    unsigned int partition_num = polar_setting.get_partition_num();
    string neuron_approx_type = polar_setting.get_neuron_approx_type();
    string remainder_type = polar_setting.get_remainder_type();
    
    Interval tmRange;
    input.intEval(tmRange, domain);
    
    UnivariatePolynomial<Real> up = gen_bern_poly("ReLU", tmRange, bernstein_order);

    double error = gen_bern_err_by_sample(up, "ReLU", tmRange, partition_num);

    Interval rem(-error, error);

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
}

void Neuron::affine_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const
{
    unsigned int taylor_order = polar_setting.get_taylor_order();
    unsigned int bernstein_order = polar_setting.get_bernstein_order();
    unsigned int partition_num = polar_setting.get_partition_num();
    string neuron_approx_type = polar_setting.get_neuron_approx_type();
    string remainder_type = polar_setting.get_remainder_type();
    
    Interval tmRange;
    input.intEval(tmRange, domain);
    
    vector<Real> coe;
    coe.push_back(Real(0.0));
    coe.push_back(Real(1.0));
    UnivariatePolynomial<Real> up(coe);

    Interval rem(0);

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
}

