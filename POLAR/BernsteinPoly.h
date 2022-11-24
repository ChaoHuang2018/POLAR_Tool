
#include "../flowstar/flowstar-toolbox/UnivariatePolynomial.h"

using namespace flowstar;
using namespace std;

long factorial(int n);

long combo(int n, int m);

double relu(double x);

double sigmoid(double x);

double tanh(double x);

double relu_lips(Interval &intv);

double sigmoid_de(double x);

double sigmoid_lips(Interval &intv);

double tanh_de(double x);

double tanh_lips(Interval &intv);

void gen_bern_poly(UnivariatePolynomial<Real> &result, string act, Interval &intv, int d);

double gen_bern_err(string act, Interval intv, int degree);

double gen_bern_err_by_sample(UnivariatePolynomial<Real> &berns, string act, Interval &intv, int partition);
