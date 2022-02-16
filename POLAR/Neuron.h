#ifndef NEURON_H_
#define NEURON_H_

#include "../flowstar-toolbox/Continuous.h"
#include "PolarSetting.h"
#include "BernsteinPoly.h"

using namespace flowstar;
using namespace std;

class Neuron
{
protected:
    string activation;

    void sigmoid_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const;
    void relu_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const;
    void tanh_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const;
    void affine_taylor(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const;

public:
    Neuron();
    Neuron(string act);
    
    void taylor_model_approx(TaylorModel<Real> &result, TaylorModel<Real> &input, const std::vector<Interval> &domain, PolarSetting &polar_setting, const Computational_Setting &setting) const;
    
};


#endif /* NEURON_H_ */
