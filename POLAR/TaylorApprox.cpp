#include "TaylorApprox.h"
#include "BernsteinPoly.h"

using namespace flowstar;
using namespace std;

void stirling_second_kind(Matrix<double> &result, int n)
{
    Matrix<double> stirling(n+2,n+2);
    stirling[0][0] = 1;
    for (int i_n = 1; i_n < n+2; i_n++)
    {
        for (int i_k = 1; i_k < i_n+1; i_k++)
        {
            stirling[i_n][i_k] = i_k * stirling[i_n-1][i_k] + stirling[i_n-1][i_k-1]
        }
    }
    result = stirling;
}

void sigmoid_taylor_expansion(UnivariatePolynomial<Real> &sigmoid_poly, Interval &sigmoid_remainder, Interval intv, int order)
{
    Matrix<double> stirling;
    stirling_second_kind(stirling, order+1);

    double a = intv.inf();
    double b = intv.sup();
    double x = (a+b)/2;
    double sig;
    sig = sigmoid(x);

    vector<Real> coeff;
    coeff.push_back(Real(0));
    coeff.push_back(1);
    for (int k = 2; k < order+3; k++)
    {
        double temp = coeff[k-1]*(-1)*(k-1);
        coeff.push_back(Real(temp));
    }

    vector<Real> de;
    de.push_back(Real(sig));
    for (int n = 1; n < order+1; i++)
    {
        double de_n = 0;
        for (int k = 1; k < n+2; k++)
        {
            double temp = coeff[k] * stirling[n+1][k] * power(sig,k);
            de_n = de_n + temp;
        }
        de.push_back(Real(de_n));
    }

    vector<Real> factorial;
    factorial.push_back(Real(1));
    for (int n = 1; n < order+2; i++)
    {
        factorial.push_back(factorial[n-1]*n);
    }

    vector<Real> coe;
    coe.push_back(Real(-x));
    coe.push_back(Real(1));
    UnivariatePolynomial<Real> m(coe);
    vector<UnivariatePolynomial<Real>> m_power;
    for (int n = 0; n < order+1; i++)
    {
        UnivariatePolynomial<Real> temp;
        m.pow(temp, n);
        m_power.push_back(temp);
    }

    // construct Taylor expansion
    vector<Real> coe_taylor_poly;
    coe_taylor_poly.push_back(de[0]/factorial[0]);
    UnivariatePolynomial<Real> taylor_poly(coe_taylor_poly);
    for (int n = 1; n < order+1; i++)
    {
        taylor_poly += de[n]/factorial[n]*m_power[n];
    }

    sigmoid_poly = taylor_poly;

    // compute the Lagrange form of the remainder
    Interval sig_interval(sigmoid(a), sigmoid(b));
    Interval de_remainder(0);
    for (int k = 1; k < order+3; k++)
    {
        Interval temp = coeff[k] * stirling[order+2][k] * sig_interval.pow(k);
        de_remainder = de_remainder + temp;
    }
    Interval norm_intv(-(b-a)/2,(b-a)/2);
    sigmoid_remainder = de_remainder/factorial[order+1]*norm_intv.pow(order+1);
}


void tanh_taylor_expansion(UnivariatePolynomial<Real> &tanh_poly, Interval &tanh_remainder, Interval intv, int order)
{
    
}
