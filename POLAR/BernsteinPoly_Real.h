#include "../flowstar/UnivariatePolynomial.h"

using namespace flowstar;
using namespace std;

Real factorial(int n);

Real combo(int n, int m);

Real relu(Real x);

Real sigmoid(Real x);

Real tanh(Real x);

Real relu_lips(Interval &intv);

Real sigmoid_de(Real x);

Real sigmoid_lips(Interval &intv);

Real tanh_de(Real x);

Real tanh_lips(Interval &intv);

void gen_bern_poly(UnivariatePolynomial<Real> &result, string act, Interval intv, int d);

double gen_bern_err(string act, Interval intv, int degree);

Real gen_bern_err_by_sample(UnivariatePolynomial<Real> &berns, string act, Interval intv, int partition);
