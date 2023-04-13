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
	/**
	std::ifstream inputs;
    inputs.open("./flowstar_1step_inputs");
    string argv[23];
    std::string line;
	int i = 1;
    while (std::getline(inputs, line))
    {
        argv[i++] = line;
    }
    
	for(size_t i = 0; i < sizeof(argv); i++) {
		cout << i << " " << argv[i] << endl;
	}

	
	
	return 0;
	**/
	string argv[22];
	size_t pos = 0;
	std::string s = argvs[1];
	//cout << "arguments: " << s << endl;
	std::string delimiter = "::";
	
	
	int i = 1;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		argv[i++] = s.substr(0, pos).c_str();
		s.erase(0, pos + delimiter.length());
		if(i == 22) break;
	}
	
	string benchmark_name = argv[2];
	// Declaration of the state variables.
	unsigned int numVars = 9;
	
	intervalNumPrecision = 600;
	Variables vars;
	 

	int x0_id = vars.declareVar("x0"); // v_set
	int x1_id = vars.declareVar("x1");	// T_gap
	int x2_id = vars.declareVar("x2");	// x_lead
	int x3_id = vars.declareVar("x3");	// x_ego
	int x4_id = vars.declareVar("x4"); // v_lead
	int x5_id = vars.declareVar("x5");	// v_ego
	int x6_id = vars.declareVar("x6");	// gamma_lead
	int x7_id = vars.declareVar("x7");	// gamma_ego
	int u0_id = vars.declareVar("u0");	// a_ego

	int domainDim = numVars + 1;
	/*
	// Define the continuous dynamics.
	Expression<Real> deriv_x0("0", vars); //  
	Expression<Real> deriv_x1("0", vars); 
	Expression<Real> deriv_x2("x4", vars);
	Expression<Real> deriv_x3("x5", vars);
	Expression<Real> deriv_x4("x6", vars);
	Expression<Real> deriv_x5("x7", vars);
	Expression<Real> deriv_x6("-2 * 2 - 2 * x6 - 0.0001 * x4 * x4", vars);
	Expression<Real> deriv_x7("2 * u0 - 2 * x7 - 0.0001 * x5 * x5", vars);
	Expression<Real> deriv_u0("0", vars);

	vector<Expression<Real>> ode_rhs(numVars);
	ode_rhs[x0_id] = deriv_x0;
	ode_rhs[x1_id] = deriv_x1;
	ode_rhs[x2_id] = deriv_x2;
	ode_rhs[x3_id] = deriv_x3;
	ode_rhs[x4_id] = deriv_x4;
	ode_rhs[x5_id] = deriv_x5;
	ode_rhs[x6_id] = deriv_x6;
	ode_rhs[x7_id] = deriv_x7;
	ode_rhs[u0_id] = deriv_u0;

	Deterministic_Continuous_Dynamics dynamics(ode_rhs);
	*/
	// Define the continuous dynamics.
	ODE<Real> dynamics({"0","0","x4", "x5","x6","x7","-2 * 2 - 2 * x6 - 0.0001 * x4 * x4","2 * u0 - 2 * x7 - 0.0001 * x5 * x5","0"}, vars);
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[1]);
	
	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order);

	// time horizon for a single control step
	//setting.setTime(0.1);

	// cutoff threshold
	setting.setCutoffThreshold(1e-8);

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
	/*
	for(int i = 4; i <= 21; i++) {
		cout << i << "hehehe" << stod(argv[i]) << "hahaha" << endl;	
	}
	*/
	Interval 
		init_x0(stod(argv[4]), stod(argv[5])), 
		init_x1(stod(argv[6]), stod(argv[7])), 
		init_x2(stod(argv[8]), stod(argv[9])), 
		init_x3(stod(argv[10]), stod(argv[11])), 
		init_x4(stod(argv[12]), stod(argv[13])), 
		init_x5(stod(argv[14]), stod(argv[15])), 
		init_x6(stod(argv[16]), stod(argv[17])), 
		init_x7(stod(argv[18]), stod(argv[19])),
		init_u0(stod(argv[20]), stod(argv[21])); // w=0.05
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_x6);
	X0.push_back(init_x7);
	X0.push_back(init_u0);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 500);

	// no unsafe set
	vector<Constraint> unsafeSet;
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// Always using symbolic remainder
	dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);
	//dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);

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
	Constraint c1("x5 - x4", vars);		// x0 <= 0.2
 
	targetSet.push_back(c1);
 

	bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
	string reach_result;

	cout.precision(17);
	Interval box;
	for(int i = 0; i < 8; i++) {
		initial_set.tmvPre.tms[vars.getIDForVar("x" + std::to_string(i))].intEval(box, initial_set.domain);
		cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
	}

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);

    const string dir_name =  "./outputs/"  + benchmark_name + "_crown_flowstar";
    char* c = const_cast<char*>(dir_name.c_str());

	int mkres = mkdir(c, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
    plot_setting.setOutputDims("x4", "x5");
    //plot_setting.plot_2D_octagon_MATLAB(c, "/" + to_string(stoi(argv[3])), result);
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", "step_"  + to_string(stoi(argv[3])) + "_"  + to_string(1), result.tmv_flowpipes, setting);
    


	return 0;
}

