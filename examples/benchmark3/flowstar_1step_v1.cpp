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
argv[6]: u_min
argv[7]: u_max
argv[8]: step
argv[9]: net name


*/

int main(int argc, char *argvs[])
{
	string argv[10];
	size_t pos = 0;
	std::string s = argvs[1];
	//cout << "arguments: " << s << endl;
	std::string delimiter = "::";
	
	int i = 1;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		argv[i++] = s.substr(0, pos).c_str();
		s.erase(0, pos + delimiter.length());
		if(i == sizeof(argv)) break;
	}
 
	string benchmark_name = argv[2];
	// Declaration of the state variables.
	unsigned int numVars = 3;
	
    intervalNumPrecision = 600;

	Variables vars;


	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;
	/*
	// Define the continuous dynamics.
	Expression<Real> deriv_x0("-x0*(0.1+(x0+x1)^2)", vars); // theta_r = 0
	Expression<Real> deriv_x1("(u+x0)*(0.1+(x0+x1)^2)", vars);
	Expression<Real> deriv_u("0", vars);

	vector<Expression<Real> > ode_rhs(numVars);
	ode_rhs[x0_id] = deriv_x0;
	ode_rhs[x1_id] = deriv_x1;
	ode_rhs[u_id] = deriv_u;

	Deterministic_Continuous_Dynamics dynamics(ode_rhs);
	*/
	// Define the continuous dynamics.
	ODE<Real> dynamics({"-x0*(0.1+(x0+x1)^2)","(u+x0)*(0.1+(x0+x1)^2)","0"}, vars);
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[1]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.02, order);

	// time horizon for a single control step
	//setting.setTime(0.1);

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
	
	// double w = stod(argv[1]);
	int steps = 1;
	for(int i = 4; i <= 9; i++) {
		cout << i << "hehehe" << argv[i] << "hahaha" << endl;
		cout << i << "hehehe" << stod(argv[i]) << "hahaha" << endl;	
	}
	 */
	Interval init_x0(stod(argv[4]), stod(argv[5])), init_x1(stod(argv[6]), stod(argv[7])), init_u(stod(argv[8]), stod(argv[9])); // w=0.05
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_u);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 1000);

	// no unsafe set
	vector<Constraint> unsafeSet;
	// no unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// Always using symbolic remainder
	//dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);
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



	vector<Constraint> targetSet;
	Constraint c1("x0 - 0.3", vars);		// x0 <= 0.3
	Constraint c2("-x0 + 0.2", vars);		// x0 >= 0.2
	Constraint c3("x1 + 0.05", vars);		// x1 <= -0.05
	Constraint c4("-x1 - 0.3", vars);		// x1 >= -0.3


	targetSet.push_back(c1);
	targetSet.push_back(c2);
	targetSet.push_back(c3);
	targetSet.push_back(c4);

	bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
	string reach_result;

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
    //plot_setting.plot_2D_octagon_MATLAB(c, "/step" + to_string(stoi(argv[3])), result);
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", "step_"  + to_string(stoi(argv[3])) + "_"  + to_string(1), result.tmv_flowpipes, setting);
    


	return 0;
}
