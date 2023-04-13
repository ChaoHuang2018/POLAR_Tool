#include "../../POLAR/NeuralNetwork.h"
#include "../../flowstar/flowstar-toolbox/Discrete.h"
#include <chrono>
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	string net_name = argv[6];
	string benchmark_name = "mountain_car_" + net_name;
	// Declaration of the state variables.
	unsigned int numVars = 3;

	intervalNumPrecision = 200;

	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;
	
	/* Old inteface
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
	// Define the discrete dynamics.
	DDE<Real> dynamics({"x0 + x1",
			    "x1 + 0.0015 * u - 0.0025 * cos(3 * x0)",
			    "0"}, vars);
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);
	//Computational_Setting setting;

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.01, order); // the stepsize will be ignored

	// cutoff threshold
	setting.setCutoffThreshold(1e-8);

	// print out the steps
	setting.printOff();

/*	// DDE does not require a remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);
*/
	//setting.printOn();

	//setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	double w = stod(argv[1]);
	int steps = stoi(argv[2]);
	Interval init_x0(-0.515 - w, -0.515 + w), init_x1(0);
	Interval init_u(0); // w=0.05
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_u);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 100);

	// no unsafe set
	vector<Constraint> safeSet;
	//vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = "nn_"+net_name;
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-12, 1e-12);
	unsigned int bernstein_order = stoi(argv[3]);
	// unsigned int partition_num = 4000;
	unsigned int partition_num = 200;

	unsigned int if_symbo = stoi(argv[5]);

	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	double nn_total_time = 0.0, flowstar_total_time = 0.0;
	time(&start_timer);

	vector<string> state_vars;
	state_vars.push_back("x0");
	state_vars.push_back("x1");

	if (if_symbo == 0)
	{
		cout << "High order abstraction starts." << endl;
	}
	else
	{
		cout << "High order abstraction with symbolic remainder starts." << endl;
	}

	auto begin = std::chrono::high_resolution_clock::now();
	for (int iter = 0; iter < steps; ++iter)
	{
		cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		TaylorModelVec<Real> tmv_input;

		tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);


		// taylor propagation
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
		polar_setting.set_num_threads(-1);
		TaylorModelVec<Real> tmv_output;

		auto inner_begin = std::chrono::high_resolution_clock::now();

		if(if_symbo == 0){
			// not using symbolic remainder
			nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}
		else{
			// using symbolic remainder
			nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}
		auto nn_timing = std::chrono::high_resolution_clock::now();
		auto nn_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(nn_timing - inner_begin);
		seconds = nn_elapsed.count() * 1e-9;
		printf("nn time is %.2f s.\n", seconds);
		nn_total_time += seconds;


		Matrix<Interval> rm1(1, 1);
		tmv_output.Remainder(rm1);
		cout << "Neural network taylor remainder: " << rm1 << endl;



		initial_set.tmvPre.tms[u_id] = tmv_output.tms[0];

		// if(if_symbo == 0){
		// 	dynamics.reach(result, setting, initial_set, unsafeSet);
		// }
		// else{
		// 	dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);
		// }

		// Always using symbolic remainder
		dynamics.reach(result, setting, initial_set, 1, safeSet, symbolic_remainder);

		auto flowstar_timing = std::chrono::high_resolution_clock::now();
		auto flowstar_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(flowstar_timing - nn_timing);
		seconds = flowstar_elapsed.count() * 1e-9;
		printf("flowstar time is %.2f s.\n", seconds);
		flowstar_total_time += seconds;

		//dynamics.reach_sr(result, setting, initial_set, 1, symbolic_remainder, unsafeSet);

		// not using a symbolic remainder
		// dynamics.reach(result, setting, initial_set, 1, unsafeSet);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			return 1;
		}
	}
	printf("NN time: %.2fs, Flow* time: %.2fs.\n", nn_total_time, flowstar_total_time);

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
	seconds = elapsed.count() *  1e-9;
	printf("Time measured: %.3f seconds.\n", seconds);

	vector<Constraint> targetSet;
	Constraint c1("-x0 + 0.45", vars);		// x0 >= 0.2
	Constraint c2("-x1 + 0.0", vars);		// x0 >= 0.0

	targetSet.push_back(c1);
	targetSet.push_back(c2);

	string reach_result;

	bool b = result.fp_end_of_time.isInTarget(targetSet, setting);


	if(b)
	{
		reach_result = "Verification result: Yes(" + to_string(steps) + ")";
	}
	else
	{
		reach_result = "Verification result: No(" + to_string(steps) + ")";
	}
	cout << reach_result << endl;


	// time(&end_timer);
	// seconds = difftime(start_timer, end_timer);

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);
	plot_setting.setOutputDims("x0", "x1");

	int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	std::string running_time = "Running Time: " + to_string(seconds) + " seconds";

	ofstream result_output("./outputs/" + benchmark_name + "_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", benchmark_name + "_" + to_string(if_symbo), result.tmv_flowpipes, setting);
	//plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(if_symbo), result);

	return 0;
}
