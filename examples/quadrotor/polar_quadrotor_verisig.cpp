#include <unordered_map>
#include <queue>
#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

Deterministic_Continuous_Dynamics build_dynamics(Variables &vars, unsigned int numVars, unordered_map<string, int> &var_ids, int segment){
	vector<string> x1_derivs = {"-0.25+x4", "-0.25+x4", "x4", "0.25+x4"};
	vector<string> x2_derivs = {"0.25+x5", "-0.25+x5", "0.25+x5", "-0.25+x5"};

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
	return dynamics;
}

vector<TaylorModelVec<Real>> build_ctrl_lookup_table(int numVars){
	vector<TaylorModelVec<Real>> lookup_table;
	vector<vector<Real>> table = {{-0.1, -0.1, 7.81},
								  {-0.1, -0.1, 11.81},
								  {-0.1, 0.1, 7.81},
								  {-0.1, 0.1, 11.81},
								  {0.1, -0.1, 7.81},
								  {0.1, -0.1, 11.81},
								  {0.1, 0.1, 7.81},
								  {0.1, 0.1, 11.81}};
	for(vector<Real> row: table){
		lookup_table.push_back(TaylorModelVec<Real>(row, numVars));
	}
	return lookup_table;
}

bool maybe_the_largest(const TaylorModelVec<Real> &nn_outputs, int i, const vector<Interval> &domain){
	// determine if it is possible that the i-th output of the nn is the largest
	for(int j=0; j<nn_outputs.tms.size(); j++){
		if(j == i) continue;
		TaylorModel<Real> tm = nn_outputs.tms[j] - nn_outputs.tms[i];
		Interval box;
		tm.intEval(box, domain);
		if(box.inf() > 0) return false; 
	}
	return true;
}

vector<Interval> copy_domain(const vector<Interval> &domain){
	vector<Interval> new_domain;
	for(Interval itvl: domain){
		Interval new_itvl(itvl);
		new_domain.push_back(new_itvl);
	}
	return new_domain;
}

vector<Constraint> build_guard(Variables & nn_out_vars, int i){
	// build guard constraint for the i-th nn output
	vector<string> vars = {"y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"};
	vector<Constraint> guard;
	for(int j=0; j<nn_out_vars.size(); j++){
		if(j == i) continue;
		Constraint c(vars[j] + " - " + vars[i], nn_out_vars);	// yj - yi <= 0 => yi >= yj
		guard.push_back(c);
	}
	return guard;
}

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

	vector<Deterministic_Continuous_Dynamics> dynamics;
	for(int i=0; i<4; i++) dynamics.push_back(build_dynamics(vars, numVars, var_ids, i));
	int segment = 0;
	vector<int> segment_ends = {10, 20, 25};

	// Specify the parameters for reachability computation.
	Computational_Setting setting;

	unsigned int order = stoi(argv[4]);

	double stepsize = stod(argv[1]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(stepsize, order);

	// time horizon for a single control step
	setting.setTime(0.2);

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
	int steps = stoi(argv[2]);
	Interval init_x1(-0.05, -0.025), init_x2(-0.025, 0);
	Interval init_x3(0), init_x4(0);
	Interval init_x5(0), init_x6(0);
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
	queue<Flowpipe> initial_sets;
	initial_sets.push(initial_set);

	Symbolic_Remainder symbolic_remainder(initial_set, 500);
	queue<Symbolic_Remainder> symbolic_remainders;
	symbolic_remainders.push(symbolic_remainder);

	// no unsafe set
	vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	// string nn_name = "tanh20x20_remodel";
	string nn_name = "tanh20x20";	// Original model used in Verisig
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-12, 1e-12);
	unsigned int bernstein_order = stoi(argv[3]);
	unsigned int partition_num = 4000;

	unsigned int if_symbo = stoi(argv[5]);

	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	vector<string> state_vars;
	state_vars.push_back("x1");
	state_vars.push_back("x2");
	state_vars.push_back("x3");
	state_vars.push_back("x4");
	state_vars.push_back("x5");
	state_vars.push_back("x6");

	Variables nn_out_vars;

	var_ids["y1"] = nn_out_vars.declareVar("y1");
	var_ids["y2"] = nn_out_vars.declareVar("y2");
	var_ids["y3"] = nn_out_vars.declareVar("y3");
	var_ids["y4"] = nn_out_vars.declareVar("y4");
	var_ids["y5"] = nn_out_vars.declareVar("y5");
	var_ids["y6"] = nn_out_vars.declareVar("y6");
	var_ids["y7"] = nn_out_vars.declareVar("y7");
	var_ids["y8"] = nn_out_vars.declareVar("y8");

	vector<TaylorModelVec<Real>> ctrl_lookup = build_ctrl_lookup_table(numVars);
	vector<vector<Constraint>> nn_out_guards;
	for(int i=0; i<8; i++){
		nn_out_guards.push_back(build_guard(nn_out_vars, i));
	}

	if (if_symbo == 0)
	{
		cout << "High order abstraction starts." << endl;
	}
	else
	{
		cout << "High order abstraction with symbolic remainder starts." << endl;
	}
	// return 0;

	queue<Flowpipe> mid_initial_sets;
	queue<Symbolic_Remainder> mid_symbolic_remainders;

	// perform 35 control steps
	for (int iter = 0; iter < steps; ++iter)
	{
		cout << "Step " << iter << " starts.      " << endl;
		while(!initial_sets.empty()){
			initial_set = initial_sets.front();
			initial_sets.pop();
			symbolic_remainder = symbolic_remainders.front();
			symbolic_remainders.pop();
			TaylorModelVec<Real> tmv_input;

			// tmv_input.tms.push_back(initial_set.tmvPre.tms[0]*0.2);
			// tmv_input.tms.push_back(initial_set.tmvPre.tms[1]*0.2);
			// tmv_input.tms.push_back(initial_set.tmvPre.tms[2]*0.2);
			// tmv_input.tms.push_back(initial_set.tmvPre.tms[3]*0.1);
			// tmv_input.tms.push_back(initial_set.tmvPre.tms[4]*0.1);
			// tmv_input.tms.push_back(initial_set.tmvPre.tms[5]*0.1);

			TaylorModelVec<Real> tmv_temp;
			initial_set.compose(tmv_temp, order, cutoff_threshold);
			tmv_input.tms.push_back(tmv_temp.tms[0]*0.2);
			tmv_input.tms.push_back(tmv_temp.tms[1]*0.2);
			tmv_input.tms.push_back(tmv_temp.tms[2]*0.2);
			tmv_input.tms.push_back(tmv_temp.tms[3]*0.1);
			tmv_input.tms.push_back(tmv_temp.tms[4]*0.1);
			tmv_input.tms.push_back(tmv_temp.tms[5]*0.1);

			// taylor propagation
			PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
			TaylorModelVec<Real> tmv_output;

			if(if_symbo == 0){
				// not using symbolic remainder
				nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
			}
			else{
				// using symbolic remainder
				nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
			}

			Matrix<Interval> rm1(1, 8);
			tmv_output.Remainder(rm1);
			cout << "Neural network taylor remainder: " << rm1 << endl;

			vector<int> possible_argmax;
			for(int i=0; i<tmv_output.tms.size(); i++){
				if(maybe_the_largest(tmv_output, i, initial_set.domain)){
					possible_argmax.push_back(i);
				}
			}
			if(possible_argmax.size() < 1){
				cout<<"ERROR: non of the output are determined as the possible maximal output." << endl;
				return 1;
			}
			else if(possible_argmax.size() > 1){
				cout<<"Flowpipe will split into " << possible_argmax.size() << " new flowpipes." << endl;
				for(int i=0; i<possible_argmax.size(); i++){
					vector<Interval> domain = copy_domain(initial_set.domain);
					if(domain_contraction_int(tmv_output, domain, nn_out_guards[possible_argmax[i]], order, cutoff_threshold, setting.g_setting) == UNSAT) continue;
					cout<<"New split with the " << possible_argmax[i] << "-th control input." << endl;
					Flowpipe new_initial_set(tmv_temp, domain);
					Symbolic_Remainder new_symbolic_remainder(symbolic_remainder);
					TaylorModelVec<Real> tmv_u = ctrl_lookup[possible_argmax[i]];
					new_initial_set.tmvPre.tms[var_ids["u1"]] = tmv_u.tms[0];
					new_initial_set.tmvPre.tms[var_ids["u2"]] = tmv_u.tms[1];
					new_initial_set.tmvPre.tms[var_ids["u3"]] = tmv_u.tms[2];
					mid_initial_sets.push(new_initial_set);
					mid_symbolic_remainders.push(new_symbolic_remainder);
				}
			}
			else{
				cout<<"Pick the " << possible_argmax[0] << "-th control input." << endl;
				TaylorModelVec<Real> tmv_u = ctrl_lookup[possible_argmax[0]];
				initial_set.tmvPre.tms[var_ids["u1"]] = tmv_u.tms[0];
				initial_set.tmvPre.tms[var_ids["u2"]] = tmv_u.tms[1];
				initial_set.tmvPre.tms[var_ids["u3"]] = tmv_u.tms[2];
				mid_initial_sets.push(initial_set);
				mid_symbolic_remainders.push(symbolic_remainder);
			}
		}

		if(segment < 3 && iter >= segment_ends[segment]) segment++;

		while(!mid_initial_sets.empty()){
			initial_set = mid_initial_sets.front();
			mid_initial_sets.pop();
			symbolic_remainder = mid_symbolic_remainders.front();
			mid_symbolic_remainders.pop();
			// Always using symbolic remainder
			dynamics[segment].reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);

			if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
			{
				initial_set = result.fp_end_of_time;
				initial_sets.push(initial_set);
				symbolic_remainders.push(symbolic_remainder);
				cout << initial_sets.size() << "Flowpipe(s) derived." << endl;
				// cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
			}
			else
			{
				printf("Terminated due to too large overestimation.\n");
				return 1;
			}
		}
		cout<<"Total number of flowpipes after " << iter+1 << " steps: " << initial_sets.size() << endl;
	}


	vector<Constraint> targetSet;
	Constraint c1("x1 - 0.32", vars);		// x1 <= 0.32
	Constraint c2("-x1 - 0.32", vars);		// x1 >= -0.32
	Constraint c3("x2 - 0.32", vars);		// x2 <= 0.32
	Constraint c4("-x2 - 0.32", vars);		// x2 >= -0.32
	Constraint c5("x3 - 0.32", vars);		// x3 <= 0.32
	Constraint c6("-x3 - 0.32", vars);		// x3 >= -0.32

	targetSet.push_back(c1);
	targetSet.push_back(c2);
	targetSet.push_back(c3);
	targetSet.push_back(c4);
	targetSet.push_back(c5);
	targetSet.push_back(c6);

	bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
	string reach_result;

	if(b)
	{
		reach_result = "Verification result: Yes(" + to_string(steps) + ")";
	}
	else
	{
		reach_result = "Verification result: No(" + to_string(steps) + ")";
	}


	time(&end_timer);
	seconds = difftime(start_timer, end_timer);

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);

	const string dir_name = "./outputs/quadrotor_polar";
    char* c = const_cast<char*>(dir_name.c_str());

	int mkres = mkdir(c, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	std::string running_time = "Running Time: " + to_string(-seconds) + " seconds";

	ofstream result_output("./outputs/quadrotor_polar/quadrotor_stepsize_" + to_string(stepsize).substr(2,2) + "_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
	// plot_setting.setOutputDims("x1", "x2");
	// plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "polar_quadrotor_verisig_" + to_string(steps) + "_steps_" + to_string(if_symbo), result);

    plot_setting.setOutputDims("x1", "x2");
    plot_setting.plot_2D_octagon_MATLAB(c, "/quadrotor_xy_stepsize_" + to_string(stepsize).substr(2,2) + "_" + to_string(if_symbo), result);

	plot_setting.setOutputDims("x2", "x3");
	plot_setting.plot_2D_octagon_MATLAB(c, "/quadrotor_yz_stepsize_" + to_string(stepsize).substr(2,2) + "_" + to_string(if_symbo), result);

	return 0;
}
