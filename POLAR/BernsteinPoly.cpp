#include "BernsteinPoly.h"

using namespace flowstar;
using namespace std;

Real factorial(int n)
{
    Real fc = 1;
    for (int i = 1; i <= n; ++i)
        fc *= i;
    // cout << "fc: " << fc << endl;
    return fc;
}

Real combo(int n, int m)
{
    Real com = factorial(n) / (factorial(m) * factorial(n - m));
    return com;
}

Real relu(Real x)
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

Real sigmoid(Real x)
{
//    double result = 1.0 - 1.0 / (1.0 + exp(x));
    Real result;
    x.exp_RNDU(result);
    result.add_assign_RNDU(Real(1));
    result.rec_assign();
    result.mul_assign_RNDU(Real(-1));
    result.add_assign_RNDU(Real(1));
    return result;
}

Real tanh(Real x)
{
//    double result = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    Real temp1;
    Real temp2;
    
    Real result;
    x.exp_RNDU(temp1);
    
    x.mul_RNDU(temp2, Real(-1));
    temp2.exp_assign_RNDU();
    
    Real temp3;
    Real temp4;
    
    temp1.add_RNDU(temp3, temp2);
    temp1.sub_RNDU(temp4, temp2);
    
    temp4.div_RNDU(result, temp3);
    return result;
}

Real relu_lips(Interval &intv)
{
    Real a;
    intv.inf(a);
    Real b;
    intv.sup(b);
    if (b <= 0)
    {
        return Real(0);
    }
    else
    {
        return Real(1);
    }
}

Real sigmoid_de(Real x)
{
    Real temp1(sigmoid(x));

//    double result = temp1 * (1.0 - temp1);
    Real result;
    temp1.mul_RNDU(result, Real(-1));
    result.add_assign_RNDU(Real(1));
    result.mul_assign_RNDU(temp1);
    return result;
}

Real sigmoid_lips(Interval &intv)
{
    Real a;
    intv.inf(a);
    Real b;
    intv.sup(b);

    vector<Real> check_list;
    check_list.push_back(a);
    check_list.push_back(b);

    if ((a <= 0) && (b >= 0))
    {
        check_list.push_back(Real(0));
    }

    Real de_bound = sigmoid_de(check_list[0]);
    for (int i = 0; i < check_list.size(); i++)
    {
        //cout << sigmoid_de(check_list[i]) << endl;
        Real temp1;
        Real temp2;
        
        check_list[i].abs(temp1);
        de_bound.abs(temp2);
        if (temp1 >= temp2)
        {
            de_bound = temp1;
        }
    }
    Real lips;
    de_bound.abs(lips);
    return lips;
}

Real tanh_de(Real x)
{
//    double result = 1.0 - pow(tanh(x), 2.0);
    Real temp1(tanh(x));

//    double result = temp1 * (1.0 - temp1);
    Real result(temp1);
    result.pow_assign(2);
    result.mul_assign_RNDU(Real(-1));
    result.add_assign_RNDU(Real(1));
    
    return result;
}

Real tanh_lips(Interval &intv)
{
    Real a;
    intv.inf(a);
    Real b;
    intv.sup(b);
    
    vector<Real> check_list;
    check_list.push_back(a);
    check_list.push_back(b);

    if ((a <= 0) && (b >= 0))
    {
        check_list.push_back(Real(0));
    }

    Real de_bound = tanh_de(check_list[0]);
    for (int i = 0; i < check_list.size(); i++)
    {
//        if (abs(tanh_de(check_list[i])) >= abs(de_bound))
//        {
//            de_bound = tanh_de(check_list[i]);
//        }
        Real temp1;
        Real temp2;
        
        check_list[i].abs(temp1);
        de_bound.abs(temp2);
        if (temp1 >= temp2)
        {
            de_bound = temp1;
        }
    }
//    double lips = abs(de_bound);
    
    Real lips;
    de_bound.abs(lips);
    return lips;
}

void gen_bern_poly(UnivariatePolynomial<Real> &result, string act, Interval intv, int d)
{
    time_t start_timer;
    time_t end_timer;
    double seconds;
    time(&start_timer);

    // cout << "Interval of activation abstraction: " << intv << endl;

//    double a = intv.inf();
//    double b = intv.sup();
    Real a;
    intv.inf(a);
    Real b;
    intv.sup(b);

    vector<Real> coe_bern_poly;
    coe_bern_poly.push_back(Real(0));
    UnivariatePolynomial<Real> bern_poly(coe_bern_poly);

    Real (*fun_act)(Real);
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

//    if ((b - a <= 1e-10) || (fun_act(b) - fun_act(a) <= 1e-10))
//    {
//        result = bern_poly + fun_act((a + b) / 2);
//    }
    
    // discuss special situations
    // situation 1: point rather than interval
    if (a == b)
    {
        result = fun_act(a);
        return;
    }
    // situation 2: relu and interval does not contain 0.
    if (act == "ReLU")
    {
        if (a >= Real(0))
        {
            vector<Real> coe_temp;
            coe_temp.push_back(Real(0));
            coe_temp.push_back(Real(1));
            UnivariatePolynomial<Real> u_temp(coe_temp);
            result = u_temp;
            return;
        }
        if (b <= Real(0))
        {
            vector<Real> coe_temp;
            coe_temp.push_back(Real(0));
            UnivariatePolynomial<Real> u_temp(coe_temp);
            result = u_temp;
            return;
        }
    }

//    int d_max = 8;
//    int d_p = int(floor(d_max / log10(1.0 / (b - a))));
//    if (d_p > 0)
//    {
//        d = min(d_p, d);
//    }

    vector<Real> coe_1;
    Real temp1;
    b.sub_RNDU(temp1, a);
    temp1.rec_assign();
    
    Real temp2;
    temp1.mul_RNDU(temp2, a);
    temp2.mul_assign_RNDU(Real(-1));
   
//    coe_1.push_back(Real(-1.0 * a / (b - a)));
//    coe_1.push_back(Real(1.0 / (b - a)));
    coe_1.push_back(temp2);
    coe_1.push_back(temp1);
    UnivariatePolynomial<Real> m(coe_1);
    // cout << "m: " << m << endl;

    vector<Real> coe_2;
    Real temp3;
    temp1.mul_RNDU(temp3, b);
    
    Real temp4;
    temp1.mul_RNDU(temp4, Real(-1));
    
//    coe_2.push_back(Real(1.0 * b / (b - a)));
//    coe_2.push_back(Real(-1.0 * 1 / (b - a)));
    coe_2.push_back(temp3);
    coe_2.push_back(temp4);
    UnivariatePolynomial<Real> n(coe_2);
    // cout << "n: " << n << endl;

    for (int v = 0; v <= d; v++)
    {
        // coef
        Real c = combo(d, v);

        // sample value
//        double point = a + 1.0 * (b - a) / d * v;
        Real point = a + Real(1.0) * (b - a) / Real(d) * Real(v);
        
        // cout << "point: " << point << endl;
        Real f_value;
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
        bern_poly += mono_1 * mono_2 * c * f_value;
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
    Real lips;
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
//    return 1.0 / (2 * sqrt(degree)) * lips * intv.width();
    return 1.0;
}

Real gen_bern_err_by_sample(UnivariatePolynomial<Real> &berns, string act, Interval intv, int partition)
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

    Real a;
    intv.inf(a);
    Real b;
    intv.sup(b);

    Real lips;
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

    Real (*fun_act)(Real);
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

//    if (b - a <= 1e-12)
//    {
//        cout << max(fun_act((a + b) / 2) - fun_act(a), fun_act(b) - fun_act((a + b) / 2)) << endl;
//        return max(fun_act((a + b) / 2) - fun_act(a), fun_act(b) - fun_act((a + b) / 2));
//        return 1e-12 * 0.25;
//        return 0;
//    }
    
    // discuss special situations
    // situation 1: point rather than interval
    if (a == b)
    {
        return Real(0);
    }
    
    // situation 2: relu and interval does not contain 0.
    if (act == "ReLU")
    {
        if (a >= Real(0))
        {
            return Real(0);
        }
        if (b <= Real(0))
        {
            return Real(0);
        }
    }

    // for all the Lipschitz continuous activation function
    Real sample_diff = 0;
    for (int i = 0; i < partition; i++)
    {
//        double point = a + 1.0 * (b - a) / partition * (i + 0.5);
        Real point = a + Real(1.0) * (b - a) / Real(partition) * (Real(i) + Real(0.5));

        Real berns_value;
        time(&start_timer);
        berns.evaluate(berns_value, point);
        time(&end_timer1);
        seconds = -difftime(start_timer, end_timer1);
        berns_time.push_back(seconds);
        Real fun_value = fun_act(point);
        time(&end_timer2);
        seconds = -difftime(end_timer1, end_timer2);
        act_time.push_back(seconds);

        Real temp_diff = fun_act(point) - berns_value;
        temp_diff.abs_assign();

        if (temp_diff > sample_diff)
        {
            sample_diff = temp_diff;
        }
    }
//    cout << "Sample error: " << sample_diff << endl;

    double total_berns_time = 0.0;
    double total_act_time = 0.0;
    for (int i = 0; i < partition; i++)
    {
        total_berns_time += berns_time[i];
        total_act_time += act_time[i];
    }
    // cout << "average berns evaluation time: " << total_berns_time / (partition * 1.0) << " seconds" << endl;
    // cout << "average activation evaluation time: " << total_act_time / (partition * 1.0) << " seconds" << endl;

    Real width;
    intv.width(width);
    Real overhead = Real(1.0) * lips * width / Real(partition);

    time(&end_timer);
    seconds = -difftime(start_timer0, end_timer);
    // cout << "Berns err time: " << seconds << " seconds" << endl;
//    cout << "Interval: " << intv << endl;
//    cout << "Approximation error: " << overhead + sample_diff << endl;
    
    return overhead + sample_diff;
}

