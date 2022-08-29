#include "BernsteinPoly.h"

using namespace flowstar;
using namespace std;

long factorial(int n)
{
    long fc = 1;
    for (int i = 1; i <= n; ++i)
        fc *= i;
    // cout << "fc: " << fc << endl;
    return fc;
}

long combo(int n, int m)
{
    long com = factorial(n) / (factorial(m) * factorial(n - m));
    return com;
}

double relu(double x)
{
    if (x >= 0)
    {
        return x;
    }
    else
    {
        return 0;
    }
}

double sigmoid(double x)
{
    double result = 1.0 - 1.0 / (1.0 + exp(x));
    return result;
}

double tanh(double x)
{
    double result = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    return result;
}

double relu_lips(Interval &intv)
{
    double a = intv.inf();
    double b = intv.sup();
    if (b <= 0)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

double sigmoid_de(double x)
{
    double temp1 = sigmoid(x);

    double result = temp1 * (1.0 - temp1);

    return result;
}

double sigmoid_lips(Interval &intv)
{
    double a = intv.inf();
    double b = intv.sup();

    vector<double> check_list;
    check_list.push_back(a);
    check_list.push_back(b);

    if ((a <= 0) && (b >= 0))
    {
        check_list.push_back(0);
    }

    double de_bound = sigmoid_de(check_list[0]);
    for (int i = 0; i < check_list.size(); i++)
    {
        //cout << sigmoid_de(check_list[i]) << endl;
        if (abs(sigmoid_de(check_list[i])) >= abs(de_bound))
        {
            de_bound = sigmoid_de(check_list[i]);
        }
    }
    double lips = abs(de_bound);
    return lips;
}

double tanh_de(double x)
{
    double result = 1.0 - pow(tanh(x), 2.0);
    return result;
}

double tanh_lips(Interval &intv)
{
    double a = intv.inf();
    double b = intv.sup();
    vector<double> check_list;
    check_list.push_back(a);
    check_list.push_back(b);

    if ((a <= 0) && (b >= 0))
    {
        check_list.push_back(0);
    }

    double de_bound = tanh_de(check_list[0]);
    for (int i = 0; i < check_list.size(); i++)
    {
        if (abs(tanh_de(check_list[i])) >= abs(de_bound))
        {
            de_bound = tanh_de(check_list[i]);
        }
    }
    double lips = abs(de_bound);
    return lips;
}

void gen_bern_poly(UnivariatePolynomial<Real> &result, string act, Interval &intv, int d)
{
    time_t start_timer;
    time_t end_timer;
    double seconds;
    time(&start_timer);

    // cout << "Interval of activation abstraction: " << intv << endl;

    double a = intv.inf();
    double b = intv.sup();
    double width = intv.width();

    vector<Real> coe_bern_poly;
    coe_bern_poly.push_back(0);
    UnivariatePolynomial<Real> bern_poly(coe_bern_poly);

    double (*fun_act)(double);
    if (act == "ReLU")
    {
        fun_act = relu;
    }
    if (act == "sigmoid")
    {
        fun_act = sigmoid;
    }
    if (act == "tanh")
    {
        fun_act = tanh;
    }

    if (a == b)
    {
        result = fun_act(a);
        return;
    }
    // situation 2: relu and interval does not contain 0.
    if (act == "ReLU")
    {
        if (a >= 0)
        {
            vector<Real> coe_temp;
            coe_temp.push_back(Real(0));
            coe_temp.push_back(Real(1));
            UnivariatePolynomial<Real> u_temp(coe_temp);
            result = u_temp;
            return;
        }
        if (b <= 0)
        {
            vector<Real> coe_temp;
            coe_temp.push_back(Real(0));
            UnivariatePolynomial<Real> u_temp(coe_temp);
            result = u_temp;
            return;
        }
    }

    if ((width <= 1e-10) || (fun_act(b) - fun_act(a) <= 1e-10))
    {
        bern_poly = bern_poly + (fun_act(b) + fun_act(a)) * 0.5;
        result = bern_poly;
        return;
    }

    vector<Real> coe_1;
    coe_1.push_back(Real(-1.0 * a / width));
    coe_1.push_back(Real(1.0 / width));
    UnivariatePolynomial<Real> m(coe_1);
    // cout << "m: " << m << endl;

    vector<Real> coe_2;
    coe_2.push_back(Real(1.0 * b / width));
    coe_2.push_back(Real(-1.0 * 1 / width));
    UnivariatePolynomial<Real> n(coe_2);
    // cout << "n: " << n << endl;

    for (int v = 0; v <= d; v++)
    {
        // coef
        Real c = combo(d, v);

        // sample value
        double point = a + 1.0 * width / d * v;
        // cout << "point: " << point << endl;
        double f_value;
        if (act == "ReLU")
        {
            f_value = relu(point);
        }
        if (act == "sigmoid")
        {
            f_value = sigmoid(point);
        }
        if (act == "tanh")
        {
            f_value = tanh(point);
        }
        // cout << f_value << endl;

        // for monomial 1
        UnivariatePolynomial<Real> mono_1;
        m.pow(mono_1, v);
        // cout << "v: " << v << ", " << "mono_1: " << mono_1 << endl;

        // for monomial 2
        UnivariatePolynomial<Real> mono_2;
        n.pow(mono_2, d - v);
        // cout << "v: " << v << ", " << "mono_2: " << mono_2 << endl;

//        UnivariatePolynomial<Real> temp;
//        temp = mono_1 * mono_2 * Real(c) * Real(f_value);
        // cout << temp << endl
        //      << endl
        //      << endl;
//        Real mon_value;
//        temp.evaluate(mon_value, Real(2.98));
        // cout << mon_value << endl
        //      << endl
        //      << endl;
        bern_poly += mono_1 * mono_2 * c * Real(f_value);
    }
    if (bern_poly.coefficients.size() == 0)
    {
        bern_poly.coefficients.push_back(Real(0));
    }

    time(&end_timer);
    seconds = -difftime(start_timer, end_timer);
    // cout << "Berns generation time: " << seconds << " seconds" << endl;

//     cout << "Interval: " << intv << endl;
//     cout << "Bernstein Polynomial: " << bern_poly << endl;

    result = bern_poly;
}

double gen_bern_err(string act, Interval intv, int degree)
{
    double lips;
    if (act == "ReLU")
    {
        lips = relu_lips(intv);
    }
    if (act == "sigmoid")
    {
        lips = sigmoid_lips(intv);
    }
    if (act == "tanh")
    {
        lips = tanh_lips(intv);
    }
    // use the default one, for all the Lipschitz continuous activation function
    // cout << lips << endl;
    return 1.0 / (2 * sqrt(degree)) * lips * intv.width();
}

double gen_bern_err_by_sample(UnivariatePolynomial<Real> &berns, string act, Interval &intv, int partition)
{
    time_t start_timer0;
    time_t start_timer;
    time_t end_timer;
    time_t end_timer1;
    time_t end_timer2;
    double seconds;
    vector<double> berns_time;
    vector<double> act_time;

    time(&start_timer0);

    double a = intv.inf();
    double b = intv.sup();
    double width = intv.width();

    double lips;
    if (act == "ReLU")
    {
        lips = relu_lips(intv);
    }
    if (act == "sigmoid")
    {
        lips = sigmoid_lips(intv);
    }
    if (act == "tanh")
    {
        lips = tanh_lips(intv);
    }

    double (*fun_act)(double);
    if (act == "ReLU")
    {
        fun_act = relu;
    }
    if (act == "sigmoid")
    {
        fun_act = sigmoid;
    }
    if (act == "tanh")
    {
        fun_act = tanh;
    }

    // discuss special situations
    // situation 1: point rather than interval
    if (a == b)
    {
        return 0;
    }
    
    // situation 2: relu and interval does not contain 0.
    if (act == "ReLU")
    {
        if (a >= 0)
        {
            return 0;
        }
        if (b <= 0)
        {
            return 0;
        }
    }
    
    if ((width <= 1e-10) || (fun_act(b) - fun_act(a) <= 1e-10))
    {
        return (fun_act(b) - fun_act(a)) * 0.5;
    }

    // for all the Lipschitz continuous activation function
    double sample_diff = 0;
    for (int i = 0; i < partition; i++)
    {
        double point = a + 1.0 * width / partition * (i + 0.5);

        Real berns_value;
        time(&start_timer);
        berns.evaluate(berns_value, Real(point));
        time(&end_timer1);
        seconds = -difftime(start_timer, end_timer1);
        berns_time.push_back(seconds);
        double fun_value = fun_act(point);
        time(&end_timer2);
        seconds = -difftime(end_timer1, end_timer2);
        act_time.push_back(seconds);

        double temp_diff = abs(fun_act(point) - berns_value.toDouble());

        if (temp_diff > sample_diff)
        {
            sample_diff = temp_diff;
        }
    }

    double total_berns_time = 0.0;
    double total_act_time = 0.0;
    for (int i = 0; i < partition; i++)
    {
        total_berns_time += berns_time[i];
        total_act_time += act_time[i];
    }
    // cout << "average berns evaluation time: " << total_berns_time / (partition * 1.0) << " seconds" << endl;
    // cout << "average activation evaluation time: " << total_act_time / (partition * 1.0) << " seconds" << endl;

    double overhead = 1.0 * lips * width / partition;

    time(&end_timer);
    seconds = -difftime(start_timer0, end_timer);
    // cout << "Berns err time: " << seconds << " seconds" << endl;
//     cout << "Approximation error: " << overhead + sample_diff << endl;

    return overhead + sample_diff;
}
