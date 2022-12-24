#include "../../POLAR/NNCS.h"
//#include "../../flowstar/flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	
	Variables vars;

	/*
	 * The variables should be declared in the following way:
	 * 1st group: Ordered observable state variables which will be used as input of controllers.
	 * 2nd group: Control variables which are only updated by the controller.
	 * 3rd group: State variables which are not observable.
	 */

	// input of the controller
	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");

	// output of the controller
	int u_id = vars.declareVar("u");

	// the time variable t is not used by the controller
	int t_id = vars.declareVar("t");    // time t	

	// Define the continuous dynamics.
	vector<string> derivatives = {"-x0*(0.1+(x0+x1)^2)",
								"(u+x0)*(0.1+(x0+x1)^2)",
								"0","1"};

	// create the neural network object for the controller
	string net_name = argv[6];
	string benchmark_name = "nncs_reachnn_benchmark_3_" + net_name;
	string nn_name = "nn_3_"+net_name;
	NeuralNetwork nn_controller(nn_name);

	// create the NNCS object
	NNCS<Real> system(vars, 0.1, derivatives, nn_controller);


	// Flow* reachability setting
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	//setting.setFixedStepsize(stod(argv[7]), order);
	setting.setFixedStepsize(0.1, order);
	setting.setCutoffThreshold(1e-10);
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(vars.size(), I);
	setting.setRemainderEstimation(remainder_estimation);


	// POLAR setting
	unsigned int taylor_order = order;	// same as the flowpipe order, but it is not necessary
	unsigned int bernstein_order = stoi(argv[3]);
	//unsigned int partition_num = 4000;
	unsigned int partition_num = 10;
	unsigned int if_symbo = stoi(argv[5]);
	
	PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
	if(if_symbo == 1){
			// not using symbolic remainder
			polar_setting.set_remainder_type("Symbolic");
			polar_setting.symb_rem = if_symbo;
		}

	// initial set
	double w = stod(argv[1]);

	// define the initial set which is a box
	Interval init_x0(0.85 - w, 0.85 + w), init_x1(0.45 - w, 0.45 + w);

	// the initial values of the rest of the state variables are 0
	vector<Interval> box(vars.size());
	box[x0_id] = init_x0;
	box[x1_id] = init_x1;

	// translate the initial set to a flowpipe
	Flowpipe initialSet(box);

	// unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// run the reachability computation
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	int n = stoi(argv[2]); // total number of control steps

	//Symbolic_Remainder sr(initialSet, 1000);
	Symbolic_Remainder sr(initialSet, 100);

	system.reach(result, initialSet, n, setting, polar_setting, safeSet, sr);
	
	// end box or target set
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

	if(b)
	{
		reach_result = "Verification result: Yes(" + to_string(n) + ")";
	}
	else
	{
		reach_result = "Verification result: No(" + to_string(n) + ")";
	}

	// time cost
	time(&end_timer);
	seconds = difftime(start_timer, end_timer);
	printf("time cost: %lf\n", -seconds);
	std::string running_time ="Running Time: %lf\n" + to_string(-seconds) + " seconds";

	// create a subdir named outputs to save result
	int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	ofstream result_output("./outputs/" + benchmark_name + "_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}

	// plot the flowpipes
	result.transformToTaylorModels(setting);
	Plot_Setting plot_setting(vars);
	plot_setting.printOn();
	plot_setting.setOutputDims("x0", "x1");
	plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(n) + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);

	//plot_setting.setOutputDims("t", "x1");
	//plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(n) + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);

	return 0;
}
