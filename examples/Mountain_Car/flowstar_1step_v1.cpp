#include "../../POLAR/NeuralNetwork.h"
#include "../../flowstar/flowstar-toolbox/Discrete.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

/* ******** Argument list ********

argv[1]: order
argv[2]: x0_min
argv[3]: x0_max
argv[4]: x1_min
argv[5]: x1_max
argv[6]: u_min
argv[7]: u_max
argv[8]: step
argv[9]: net name


*/

int main(int argc, char *argv[])
{
	string net_name = argv[9];
	string benchmark_name = "mountain_car" + net_name;
	// Declaration of the state variables.
	unsigned int numVars = 3;
	
    intervalNumPrecision = 600;

	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;

	/*
	// Define the discrete dynamics.
    // x0 is the position of the mountain car, x1 is the speed of the mountain car.
	Expression<Interval> deriv_x0("x0 + x1", vars); // Discrete: Next_x0 = x0 + x1
	Expression<Interval> deriv_x1("x1 + 0.0015 * u - 0.0025 * cos(3 * x0)", vars); // Discrete: Next_x1 = x1 + 0.0015 * u - 0.0025 * cos(3 * x0)
	Expression<Interval> deriv_u("u", vars);

	vector<Expression<Interval> > dde_rhs(numVars);
	dde_rhs[x0_id] = deriv_x0;
	dde_rhs[x1_id] = deriv_x1;
	dde_rhs[u_id] = deriv_u;


	Nonlinear_Discrete_Dynamics dynamics(dde_rhs);
	*/
	ODE<Real> dynamics({
		"x0 + x1",
		"x1 + 0.0015 * u - 0.0025 * cos(3 * x0)",
		"0"}, vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[1]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.01, order);

	// cutoff threshold
	setting.setCutoffThreshold(1e-10);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

	//setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	// double w = stod(argv[1]);
	int steps = 1;
	Interval init_x0(stod(argv[2]), stod(argv[3])), init_x1(stod(argv[4]), stod(argv[5])), init_u(stod(argv[6]), stod(argv[7])); // w=0.05
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_u);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 1000);

	// no unsafe set
	vector<Constraint> unsafeSet;
	vector<Constraint> safeSet;
	// result of the reachability computation
	Result_of_Reachability result;

	// Always using symbolic remainder
	//dynamics.reach_sr(result, setting, initial_set, 1, symbolic_remainder, unsafeSet);
	dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);
	
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
    plot_setting.setOutputDims("x0", "x1");
    //plot_setting.plot_2D_octagon_MATLAB(c, "/step_" + to_string(stoi(argv[8])), result);
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", "step_"  + to_string(stoi(argv[8])), result.tmv_flowpipes, setting);
    


	return 0;
}
