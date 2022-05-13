#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{

    int if_symbo = stoi(argv[1]);

    intervalNumPrecision = 800;

    Variables vars;

    unsigned int order = 4;
    unsigned int bernstein_order = 4;
    unsigned int partition_num = 1000;

    string nn_name = "motivating_nn";

    NeuralNetwork nn(nn_name);

    unsigned int numVars = 3;

    vars.declareVar("z1");
    vars.declareVar("z2");

	// Specify the parameters for reachability computation.
	Computational_Setting setting;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order);

	// time horizon for a single control step
	setting.setTime(0.2);

	// cutoff threshold
	setting.setCutoffThreshold(1e-8);

	// print out the steps
	setting.printOff();

	// remainder estimation
    vector<Interval> domain;
	Interval I(-1, 1);

    domain.push_back(I);
    domain.push_back(I);
    domain.push_back(I);

	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

	setting.prepare();

    Polynomial<Real> p1("1 - 0.5 * z1 + z2 - 0.3 * z1 * z2", vars);
    Interval r1(-0.1, 0.1);
    TaylorModel<Real> tm1(p1, r1);

    Polynomial<Real> p2("- 2 + z2 - 0.1 * z1 * z2", vars);
    Interval r2(-0.2, 0.2);
    TaylorModel<Real> tm2(p2, r2);

    TaylorModelVec<Real> tmv_input;

    tmv_input.tms.push_back(tm1);
    tmv_input.tms.push_back(tm2);

    PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
	TaylorModelVec<Real> tmv_output;

	if(if_symbo == 0){
		// not using symbolic remainder
		nn.get_output_tmv(tmv_output, tmv_input, domain, polar_setting, setting);
	}
	else{
		// using symbolic remainder
		nn.get_output_tmv_symbolic(tmv_output, tmv_input, domain, polar_setting, setting);
	}


	Matrix<Interval> rm1(1, 1);
	tmv_output.Remainder(rm1);
	cout << "Neural network taylor remainder: " << rm1 << endl;

    return 0;

}
