#include "../POLAR/BernsteinPoly.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace flowstar;

int main(int argc, char *argvs[]) {
    ofstream myfile;
    myfile.open ("bp_relu_plt.txt");
   
    Interval x(-5, 5);
    UnivariatePolynomial<Real> bp = gen_bern_poly("ReLU", x, 5);
    double a = x.inf();
    double b = x.sup();
    
    int partition = 100;
    Real berns_zero;
    bp.evaluate(berns_zero, Real(0));
    for (int i = 0; i < partition; i++)
    {
        double point = a + 1.0 * (b - a) / partition * (i + 0.5);
        Real berns_value, berns_ub, berns_lb;
        //time(&start_timer);
        bp.evaluate(berns_value, Real(point));
        berns_lb = berns_value - berns_zero;
        berns_ub = berns_value;
        berns_value = berns_value - 0.5 * berns_zero;
        //time(&end_timer1);
        //seconds = -difftime(start_timer, end_timer1);
        //berns_time.push_back(seconds);
        double fun_value = relu(point);
        myfile << point << " " << fun_value << " " << berns_lb.toDouble() << " " << berns_value.toDouble() << " " << berns_ub.toDouble() << "\n";
        //time(&end_timer2);
        //seconds = -difftime(end_timer1, end_timer2);
        //act_time.push_back(seconds);

        //double temp_diff = abs(fun_act(point) - berns_value.toDouble());

        //if (temp_diff > sample_diff)
        //{
        //    sample_diff = temp_diff;
        ///}
    }
    myfile.close();
    return 0;
}