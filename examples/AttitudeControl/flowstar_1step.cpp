#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

/* ******** Argument list ********

argv[1]: order
argv[2]: x0_min
argv[3]: x0_max
argv[4]: x1_min
argv[5]: x1_max
argv[6]: x2_min
argv[7]: x2_max
argv[8]: x3_min
argv[9]: x3_max
argv[10]: x4_min
argv[11]: x4_max
argv[12]: x5_min
argv[13]: x5_max
argv[14]: u0_min
argv[15]: u0_max
argv[16]: u1_min
argv[17]: u1_max
argv[18]: u2_min
argv[19]: u2_max
argv[20]: step
argv[21]: net name


*/

int main(int argc, char *argv[])
{
	string net_name = argv[21];
	string benchmark_name = "attitude_control_" + net_name;

    intervalNumPrecision = 600;

	// Declaration of the state variables.
	unsigned int numVars = 9;

    Variables vars;

	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int x5_id = vars.declareVar("x5");
	int u0_id = vars.declareVar("u0");
	int u1_id = vars.declareVar("u1");
	int u2_id = vars.declareVar("u2");


	int domainDim = numVars + 1;

	// Define the continuous dynamics.
	Expression<Real> deriv_x0("0.25*(u0 + x1*x2)", vars); // theta_r = 0
	Expression<Real> deriv_x1("0.5*(u1 - 3*x0*x2)", vars);
	Expression<Real> deriv_x2("u2 + 2*x0*x1", vars);
	Expression<Real> deriv_x3("0.5*x1*(x3^2 + x4^2 + x5^2 - x5) + 0.5*x2*(x3^2 + x4^2 + x4 + x5^2) + 0.5*x0*(x3^2 + x4^2 + x5^2 + 1)", vars);
	Expression<Real> deriv_x4("0.5*x0*(x3^2 + x4^2 + x5^2 + x5) + 0.5*x2*(x3^2 - x3 + x4^2 + x5^2) + 0.5*x1*(x3^2 + x4^2 + x5^2 + 1)", vars);
	Expression<Real> deriv_x5("0.5*x0*(x3^2 + x4^2 - x4 + x5^2) + 0.5*x1*(x3^2 + x3 + x4^2 + x5^2) + 0.5*x2*(x3^2 + x4^2 + x5^2 + 1)", vars);
	Expression<Real> deriv_u0("0", vars);
	Expression<Real> deriv_u1("0", vars);
	Expression<Real> deriv_u2("0", vars);

	vector<Expression<Real>> ode_rhs(numVars);
	ode_rhs[x0_id] = deriv_x0;
	ode_rhs[x1_id] = deriv_x1;
	ode_rhs[x2_id] = deriv_x2;
	ode_rhs[x3_id] = deriv_x3;
	ode_rhs[x4_id] = deriv_x4;
	ode_rhs[x5_id] = deriv_x5;
	ode_rhs[u0_id] = deriv_u0;
	ode_rhs[u1_id] = deriv_u1;
	ode_rhs[u2_id] = deriv_u2;

	Deterministic_Continuous_Dynamics dynamics(ode_rhs);

	// Specify the parameters for reachability computation.
	Computational_Setting setting;

	unsigned int order = stoi(argv[1]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order);

	// time horizon for a single control step
	setting.setTime(0.1);

	// cutoff threshold
	setting.setCutoffThreshold(1e-8);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

	setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	// double w = stod(argv[1]);
	int steps = 1;
	Interval init_x0(stod(argv[2]), stod(argv[3])), init_x1(stod(argv[4]), stod(argv[5])), init_x2(stod(argv[6]), stod(argv[7])), init_x3(stod(argv[8]), stod(argv[9])), init_x4(stod(argv[10]), stod(argv[11])), init_x5(stod(argv[12]), stod(argv[13])), init_u0(stod(argv[14]), stod(argv[15])), init_u1(stod(argv[16]), stod(argv[17])), init_u2(stod(argv[18]), stod(argv[19])); // w=0.05
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_u0);
	X0.push_back(init_u1);
	X0.push_back(init_u2);


	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 500);

	// no unsafe set
	vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// Always using symbolic remainder
	dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);

	if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
	{
		initial_set = result.fp_end_of_time;
	}
	else
	{
		printf("Terminated due to too large overestimation.\n");
		exit(1);
	}

	cout.precision(17);
	Interval box;
	initial_set.tmvPre.tms[x0_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
	initial_set.tmvPre.tms[x1_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x2_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x3_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x4_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x5_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;




	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);

    const string dir_name = "./outputs/"  + benchmark_name + "_crown_flowstar";
    char* c = const_cast<char*>(dir_name.c_str());

	int mkres = mkdir(c, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
    plot_setting.setOutputDims("x0", "x3");
    plot_setting.plot_2D_octagon_MATLAB(c, "/x0_x3_" + to_string(stoi(argv[20])), result);
    plot_setting.setOutputDims("x1", "x4");
    plot_setting.plot_2D_octagon_MATLAB(c, "/x1_x4_" + to_string(stoi(argv[20])), result);
    plot_setting.setOutputDims("x2", "x5");
    plot_setting.plot_2D_octagon_MATLAB(c, "/x2_x5_" + to_string(stoi(argv[20])), result);



	return 0;
}
