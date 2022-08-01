#include "../flowstar/UnivariatePolynomial.h"

using namespace flowstar;
using namespace std;

void stirling_second_kind(Matrix<Real> &result, int n);

void sigmoid_taylor_expansion(UnivariatePolynomial<Real> &sigmoid_poly, Interval &sigmoid_remainder, Interval intv, int order);

void tanh_taylor_expansion(UnivariatePolynomial<Real> &tanh_poly, Interval &tanh_remainder, Interval intv, int order);

