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
	string argv[24];
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
	unsigned int numVars = 10;
	
    intervalNumPrecision = 600;

	Variables vars;


	int x1_id = vars.declareVar("x1"); // x_pos
	int x2_id = vars.declareVar("x2");	// y_pos
	int x3_id = vars.declareVar("x3");	// x_vel
	int x4_id = vars.declareVar("x4");	// y_vel
	int x5_id = vars.declareVar("x5"); // ||v||
	int x6_id = vars.declareVar("x6");	// max_vel
 

	int u1_id = vars.declareVar("u1");	// fx_mean
	int u2_id = vars.declareVar("u2");	// fx_std	
	int u3_id = vars.declareVar("u3");	// fy_mean
	int u4_id = vars.declareVar("u4");	// fy_std
 

	int domainDim = numVars + 1;

	/*
	// Define the continuous dynamics.
	Expression<Real> deriv_x1("x3", vars); //  
	Expression<Real> deriv_x2("x4", vars); //   
	Expression<Real> deriv_x3("2.0 * 0.001027 * x4 + 3 * 0.001027 * 0.001027 * x1 + u1 / 12.", vars); //  
	Expression<Real> deriv_x4("-2.0 * 0.001027 * x3 + u3 / 12.", vars); //  
	Expression<Real> deriv_x5("((2.0 * 0.001027 * x4 + 3 * 0.001027 * 0.001027 * x1 + u1 / 12.) * x3 + (-2.0 * 0.001027 * x3 + u3 / 12.) * x4) / x5", vars);  
	Expression<Real> deriv_x6("2.0 * 0.001027 * (x1 * x3 + x2 * x4) / sqrt(x1 * x1 + x2 * x2)", vars);  
	//Expression<Real> deriv_x7("x2 / 1000.0", vars);  
	//Expression<Real> deriv_x8("x3 / 1000.0", vars); // deriv_x7 = u3 * x7/x6 - u2 * x8
	//Expression<Real> deriv_x9("(2.0 * 0.001027 * x3 + 3 * 0.001027 * 0.001027 * x1 + u0 / 12.) / 0.5", vars); // deriv_x8 = u3 * x8/x6 + u2 * x7
	//Expression<Real> deriv_x10("(-2.0 * 0.001027 * x2 + u1 / 12.) / 0.5", vars); // deriv_x9 = (2 * x10 * deriv(x10) + 2 * x11 * deriv(x11))/sqrt(x10*x10 + x11 * x11)
	  
	// Define the continuous dynamics according to 
	Expression<Real> deriv_u1("0", vars);
	Expression<Real> deriv_u2("0", vars);
	Expression<Real> deriv_u3("0", vars);
	Expression<Real> deriv_u4("0", vars);
	

	vector<Expression<Real> > ode_rhs(numVars);
	ode_rhs[x1_id] = deriv_x1;
	ode_rhs[x2_id] = deriv_x2;
	ode_rhs[x3_id] = deriv_x3;
	ode_rhs[x4_id] = deriv_x4;
	ode_rhs[x5_id] = deriv_x5;
	ode_rhs[x6_id] = deriv_x6;
	//ode_rhs[x8_id] = deriv_x8;
	//ode_rhs[x9_id] = deriv_x9;

	 
	ode_rhs[u1_id] = deriv_u1;
	ode_rhs[u2_id] = deriv_u2;
	ode_rhs[u3_id] = deriv_u3;
	ode_rhs[u4_id] = deriv_u4;

	Deterministic_Continuous_Dynamics dynamics(ode_rhs);
	*/
	// Define the continuous dynamics.
	ODE<Real> dynamics({
		"x3",
		"x4",
		"2.0 * 0.001027 * x4 + 3 * 0.001027 * 0.001027 * x1 + u1 / 12.",
		"-2.0 * 0.001027 * x3 + u3 / 12.",
		"((2.0 * 0.001027 * x4 + 3 * 0.001027 * 0.001027 * x1 + u1 / 12.) * x3 + (-2.0 * 0.001027 * x3 + u3 / 12.) * x4) / x5",
		"2.0 * 0.001027 * (x1 * x3 + x2 * x4) / sqrt(x1 * x1 + x2 * x2)",
		"0", "0", "0", "0"}, vars);
	
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[1]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order);

	// time horizon for a single control step
	//setting.setTime(1);

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
	Interval 
		init_x1(stod(argv[4]), stod(argv[5])), 
		init_x2(stod(argv[6]), stod(argv[7])), 
		init_x3(stod(argv[8]), stod(argv[9])), 
		init_x4(stod(argv[10]), stod(argv[11])), 
		init_x5(stod(argv[12]), stod(argv[13])),
		init_x6(stod(argv[14]), stod(argv[15])),
		//init_u1(-0.5, -0.5),
		//init_u2(0.0, 0.0),
		//init_u3(0.0, 0.0),
		//init_u4(0.0, 0.0); // w=0.05
	
		init_u1(stod(argv[16]), stod(argv[17])),
		init_u2(stod(argv[18]), stod(argv[19])),
		init_u3(stod(argv[20]), stod(argv[21])),
		init_u4(stod(argv[22]), stod(argv[23])); // w=0.05
	std::vector<Interval> X0;
 
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_x6);
	//X0.push_back(init_x6);
 
	X0.push_back(init_u1);
	X0.push_back(init_u2);
	X0.push_back(init_u3);
	X0.push_back(init_u4);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);
	
	Symbolic_Remainder symbolic_remainder(initial_set, 500);

	// no unsafe set
	vector<Constraint> unsafeSet;
	vector<Constraint> safeSet;
	//Constraint c1("x5 - x6", vars);		// x0 <= 0.2
 	//unsafeSet.push_back(c1);

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
	 

	bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
	string reach_result;

	cout.precision(17);
	Interval box;
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
	initial_set.tmvPre.tms[x6_id].intEval(box, initial_set.domain);
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
    plot_setting.setOutputDims("x5", "x6");
    //plot_setting.plot_2D_octagon_MATLAB(c, "/step" + to_string(stoi(argv[3])), result);
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", "step_"  + to_string(stoi(argv[3])) + "_"  + to_string(1), result.tmv_flowpipes, setting);
    
	

	return 0;
}

