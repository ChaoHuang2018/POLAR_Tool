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
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int x5_id = vars.declareVar("x5");

	// output of the controller
	int u0_id = vars.declareVar("u0");
	int u1_id = vars.declareVar("u1");
	int u2_id = vars.declareVar("u2");

	// the time variable t is not used by the controller
	int t_id = vars.declareVar("t");

	// define the dynamics of all variables
	vector<string> derivatives = {"0.25*(u0 + x1*x2)",
								"0.5*(u1 - 3*x0*x2)",
								"u2 + 2*x0*x1",
								"0.5*x1*(x3^2 + x4^2 + x5^2 - x5) + 0.5*x2*(x3^2 + x4^2 + x4 + x5^2) + 0.5*x0*(x3^2 + x4^2 + x5^2 + 1)",
								"0.5*x0*(x3^2 + x4^2 + x5^2 + x5) + 0.5*x2*(x3^2 - x3 + x4^2 + x5^2) + 0.5*x1*(x3^2 + x4^2 + x5^2 + 1)",
								"0.5*x0*(x3^2 + x4^2 - x4 + x5^2) + 0.5*x1*(x3^2 + x3 + x4^2 + x5^2) + 0.5*x2*(x3^2 + x4^2 + x5^2 + 1)",
								"0","0","0","1"};

	// create the neural network object for the controller
	//string net_name = argv[6];
	//string benchmark_name = "nncs_attitude_control_sigmoid" + net_name;
	string nn_name = "CLF_controller_layer_num_3";
	NeuralNetwork nn_controller(nn_name);

	// create the NNCS object
	NNCS<Real> system(vars, 0.1, derivatives, nn_controller);


	// Flow* reachability setting
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	//setting.setFixedStepsize(0.005, order);
	setting.setFixedStepsize(0.1, order);
	setting.setCutoffThreshold(1e-5);
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

	PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Symbolic");
	polar_setting.set_num_threads(-1);
	if(if_symbo == 0){
			// not using symbolic remainder
			polar_setting.set_remainder_type("Concrete");
			polar_setting.symb_rem = if_symbo;
		}

	// initial set
	double w = stod(argv[1]);

	// define the initial set which is a box
	Interval init_x0(-0.445 - w, -0.445 + w), 
			init_x1(-0.545 - w, -0.545 + w), 
			init_x2(0.655 - w, 0.655 + w), 
			init_x3(-0.745 - w, -0.745 + w), 
			init_x4(0.855 - w, 0.855 + w), 
			init_x5(-0.645 - w, -0.645 + w);

	// the initial values of the rest of the state variables are 0
	vector<Interval> box(vars.size());
	box[x0_id] = init_x0;
	box[x1_id] = init_x1;
	box[x2_id] = init_x2;
	box[x3_id] = init_x3;
	box[x4_id] = init_x4;
	box[x5_id] = init_x5;

	Flowpipe initialSet(box);

	// unsafe set
	vector<Constraint> safeSet;

	Result_of_Reachability result;

	// run the reachability computation
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	int n = stoi(argv[2]); // total number of control steps

	//Symbolic_Remainder sr(initialSet, 2000);
	Symbolic_Remainder sr(initialSet, 50);

	system.reach(result, initialSet, n, setting, polar_setting, safeSet, sr);

	// end box or target set
	vector<Interval> end_box;
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
	string reach_result;
	reach_result = "Verification result: Unknown";
	
	
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

	ofstream result_output("./outputs/nn_ac_sigmoid_"  + to_string(if_symbo) + ".txt");
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
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", "nncs_ac_sigmoid_x0_x1_" + to_string(n) + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);

	plot_setting.setOutputDims("x2", "x3");
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", "nncs_ac_sigmoid_x2_x3_" + to_string(n) + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);

	plot_setting.setOutputDims("x4", "x5");
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", "nncs_ac_sigmoid_x4_x5_" + to_string(n) + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);
	
	//plot_setting.setOutputDims("t", "x1");
	//plot_setting.plot_2D_interval_GNUPLOT("./outputs/", "nncs_ac_sigmoid_x4_x5_" + to_string(n) + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);
	
	return 0;
}
