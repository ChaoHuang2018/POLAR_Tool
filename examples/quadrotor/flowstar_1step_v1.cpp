#include <unordered_map>
#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

/* ******** Argument list ********

argv[1]: order
argv[2-13]: x1_min, x1_max, ..., x6_min, x6_max
argv[14]: u_idx
argv[15]: flowstar step size
argv[16]: step
argv[17]: flowpipe_id


*/

ODE<Real> build_dynamics(Variables &vars, unsigned int numVars, unordered_map<string, int> &var_ids, int segment){
	
	vector<string> x1_derivs = {"-0.25+x4", "-0.25+x4", "x4", "0.25+x4"};
	vector<string> x2_derivs = {"0.25+x5", "-0.25+x5", "0.25+x5", "-0.25+x5"};
	/*
	// Define the continuous dynamics.
	Expression<Real> deriv_x1(x1_derivs[segment], vars);	// Segmentation fault when the expression is "x4-0.25"
	Expression<Real> deriv_x2(x2_derivs[segment], vars);
	Expression<Real> deriv_x3("x6", vars);
	Expression<Real> deriv_x4("9.81*sin(u1)/cos(u1)", vars);
	Expression<Real> deriv_x5("-9.81*sin(u2)/cos(u2)", vars);
	Expression<Real> deriv_x6("-9.81+u3", vars);	// Segmentation fault when the expression is "u3-9.81"
	Expression<Real> deriv_u1("0", vars);
	Expression<Real> deriv_u2("0", vars);
	Expression<Real> deriv_u3("0", vars);

	vector<Expression<Real> > ode_rhs(numVars);
	ode_rhs[var_ids["x1"]] = deriv_x1;
	ode_rhs[var_ids["x2"]] = deriv_x2;
	ode_rhs[var_ids["x3"]] = deriv_x3;
	ode_rhs[var_ids["x4"]] = deriv_x4;
	ode_rhs[var_ids["x5"]] = deriv_x5;
	ode_rhs[var_ids["x6"]] = deriv_x6;
	ode_rhs[var_ids["u1"]] = deriv_u1;
	ode_rhs[var_ids["u2"]] = deriv_u2;
	ode_rhs[var_ids["u3"]] = deriv_u3;

	Deterministic_Continuous_Dynamics dynamics(ode_rhs);
	*/
	ODE<Real> dynamics({
		x1_derivs[segment],
		x2_derivs[segment],
		"x6", 
		"9.81*sin(u1)/cos(u1)",
		"-9.81*sin(u2)/cos(u2)",
		"-9.81+u3",
		"0",
		"0",
		"0"}, vars);
	
	return dynamics;
}

TaylorModelVec<Real> lookup_ctrl(int idx, int numVars){
	vector<vector<Real>> table = {{-0.1, -0.1, 7.81},
								  {-0.1, -0.1, 11.81},
								  {-0.1, 0.1, 7.81},
								  {-0.1, 0.1, 11.81},
								  {0.1, -0.1, 7.81},
								  {0.1, -0.1, 11.81},
								  {0.1, 0.1, 7.81},
								  {0.1, 0.1, 11.81}};
	vector<Real> row = table[idx];
	return TaylorModelVec<Real>(row, numVars);
}

// bool maybe_the_largest(const TaylorModelVec<Real> &nn_outputs, int i, const vector<Interval> &domain){
// 	// determine if it is possible that the i-th output of the nn is the largest
// 	for(int j=0; j<nn_outputs.tms.size(); j++){
// 		if(j == i) continue;
// 		TaylorModel<Real> tm = nn_outputs.tms[j] - nn_outputs.tms[i];
// 		Interval box;
// 		tm.intEval(box, domain);
// 		if(box.inf() > 0) return false; 
// 	}
// 	return true;
// }

// vector<Interval> copy_domain(const vector<Interval> &domain){
// 	vector<Interval> new_domain;
// 	for(Interval itvl: domain){
// 		Interval new_itvl(itvl);
// 		new_domain.push_back(new_itvl);
// 	}
// 	return new_domain;
// }

// vector<Constraint> build_guard(Variables & nn_out_vars, int i){
// 	// build guard constraint for the i-th nn output
// 	vector<string> vars = {"y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"};
// 	vector<Constraint> guard;
// 	for(int j=0; j<nn_out_vars.size(); j++){
// 		if(j == i) continue;
// 		Constraint c(vars[j] + " - " + vars[i], nn_out_vars);	// yj - yi <= 0 => yi >= yj
// 		guard.push_back(c);
// 	}
// 	return guard;
// }

int main(int argc, char *argv[])
{
	// Declaration of the state variables.
	unsigned int numVars = 9;
	
	intervalNumPrecision = 600;

	Variables vars;

	unordered_map<string, int> var_ids;
	var_ids["x1"] = vars.declareVar("x1");
	var_ids["x2"] = vars.declareVar("x2");
	var_ids["x3"] = vars.declareVar("x3");
	var_ids["x4"] = vars.declareVar("x4");
	var_ids["x5"] = vars.declareVar("x5");
	var_ids["x6"] = vars.declareVar("x6");
	var_ids["u1"] = vars.declareVar("u1");
	var_ids["u2"] = vars.declareVar("u2");
	var_ids["u3"] = vars.declareVar("u3");

	int domainDim = numVars + 1;

	//vector<Deterministic_Continuous_Dynamics> dynamics;
	vector<ODE<Real>> dynamics;
	for(int i=0; i<4; i++) dynamics.push_back(build_dynamics(vars, numVars, var_ids, i));
	int segment = 0;
	vector<int> segment_ends = {10, 20, 25};

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[1]);

	double stepsize = stod(argv[15]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(stepsize, order);

	// time horizon for a single control step
	//setting.setTime(0.2);

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
	Interval init_x1(stod(argv[2]), stod(argv[3])), init_x2(stod(argv[4]), stod(argv[5]));
	Interval init_x3(stod(argv[6]), stod(argv[7])), init_x4(stod(argv[8]), stod(argv[9]));
	Interval init_x5(stod(argv[10]), stod(argv[11])), init_x6(stod(argv[12]), stod(argv[13]));
	Interval init_u1(0), init_u2(0), init_u3(0);
	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_x6);
	X0.push_back(init_u1);
	X0.push_back(init_u2);
	X0.push_back(init_u3);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 500);

	int u_idx = stoi(argv[14]);
	TaylorModelVec<Real> tmv_u = lookup_ctrl(u_idx, numVars);
	cout << "Apply ctrl input " << u_idx << endl;
	initial_set.tmvPre.tms[var_ids["u1"]] = tmv_u.tms[0];
	initial_set.tmvPre.tms[var_ids["u2"]] = tmv_u.tms[1];
	initial_set.tmvPre.tms[var_ids["u3"]] = tmv_u.tms[2];

	// no unsafe set
	vector<Constraint> unsafeSet;
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	int iter = stoi(argv[16]);
	while(segment < 3 && iter >= segment_ends[segment]) segment++;

	int flowpipe_id = stoi(argv[17]);

	// Always using symbolic remainder
	dynamics[segment].reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);
	//dynamics[segment].reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);

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
	initial_set.tmvPre.tms[var_ids["x1"]].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
	initial_set.tmvPre.tms[var_ids["x2"]].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
	initial_set.tmvPre.tms[var_ids["x3"]].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
	initial_set.tmvPre.tms[var_ids["x4"]].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
	initial_set.tmvPre.tms[var_ids["x5"]].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
	initial_set.tmvPre.tms[var_ids["x6"]].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);

    const string dir_name = "./outputs/quadrotor_crown_flowstar";
    char* c = const_cast<char*>(dir_name.c_str());

	int mkres = mkdir(c, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
    plot_setting.setOutputDims("x1", "x2");
    //plot_setting.plot_2D_octagon_MATLAB(c, "/xy_step_" + to_string(iter) + "_" + to_string(flowpipe_id), result);
    plot_setting.plot_2D_octagon_MATLAB("./outputs/", "xy_step_"  + to_string(stoi(argv[3])) + "_"  + to_string(1), result.tmv_flowpipes, setting);
    
    plot_setting.setOutputDims("x2", "x3");
    //plot_setting.plot_2D_octagon_MATLAB(c, "/yz_step_" + to_string(iter) + "_" + to_string(flowpipe_id), result);
    plot_setting.plot_2D_octagon_MATLAB("./outputs/", "yz_step_"  + to_string(stoi(argv[3])) + "_"  + to_string(1), result.tmv_flowpipes, setting);
    


	return 0;
}

