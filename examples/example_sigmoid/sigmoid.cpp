#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{

    intervalNumPrecision = 800;

    Variables vars;

    unsigned int order = 3;
    unsigned int bernstein_order = 3;
    unsigned int partition_num = 2000;

    unsigned int numVars = 1;
    
    int x_id = vars.declareVar("x");

	// Specify the parameters for reachability computation.
	Computational_Setting setting;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order);

	// time horizon for a single control step
	setting.setTime(0.2);

	// cutoff threshold
	setting.setCutoffThreshold(1e-10);

	// print out the steps
	setting.printOff();

	// remainder estimation
    vector<Interval> domain;
	Interval I(-1, 1);

    domain.push_back(I);

	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

	setting.prepare();

    Polynomial<Real> p1("0.1*x - 0.1 * x^2", vars);
    Interval r1(-0.1, 0.1);
    TaylorModel<Real> tm_input(p1, r1);
    
//    cout << "11111111" << endl;

    PolarSetting polar_setting_berns(order, bernstein_order, partition_num, "Berns", "Concrete");
	
    TaylorModel<Real> tm_output;

    Neuron s_neuron("sigmoid");
    
    s_neuron.taylor_model_approx(tm_output, tm_input, domain, polar_setting_berns, setting, "on");
    
    tm_output.output(cout, vars);
    cout << endl;
    
//    cout << "222222" << endl;
//    cout << "Compose after Berns poly: " << tm_output.expansion << endl;
	cout << "Compose after Berns remainder: " << tm_output.remainder << endl;
    
    Polynomial<Real> p2("x", vars);
    Interval r2(0);
    TaylorModel<Real> tm_input2(p2, r2);
    
    vector<Interval> domain_x;
    Interval I_x(-0.1-0.1, 0.1+0.1);

    domain_x.push_back(I_x);
    
    
    PolarSetting polar_setting_tm(order, bernstein_order, partition_num, "Taylor", "Concrete");
    
    s_neuron.taylor_model_approx(tm_output, tm_input2, domain_x, polar_setting_tm, setting);
//    cout << "TM approximation poly: " << tm_output.expansion << endl;
    cout << "TM approximation remainder: " << tm_output.remainder << endl;
    tm_output.output(cout, vars);
    cout << endl;
    
    
    s_neuron.taylor_model_approx(tm_output, tm_input, domain, polar_setting_tm, setting);
//    cout << "Compose after TM poly: " << tm_output.expansion << endl;
    cout << "Compose after TM remainder: " << tm_output.remainder << endl;
    tm_output.output(cout, vars);
    cout << endl;

    return 0;

}
