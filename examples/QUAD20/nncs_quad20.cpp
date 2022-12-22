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
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int x5_id = vars.declareVar("x5");
	int x6_id = vars.declareVar("x6");
	int x7_id = vars.declareVar("x7");
	int x8_id = vars.declareVar("x8");
	int x9_id = vars.declareVar("x9");
	int x10_id = vars.declareVar("x10");
	int x11_id = vars.declareVar("x11");
	int x12_id = vars.declareVar("x12");
	
	// output of the controller
	int u0_id = vars.declareVar("u0");
	int u1_id = vars.declareVar("u1");
	int u2_id = vars.declareVar("u2");

	// the time variable t is not used by the controller
	int t_id = vars.declareVar("t");

	
	// define the dynamics of all variables
	vector<string> derivatives = {"cos(x8)*cos(x9)*x4 + (sin(x7)*sin(x8)*cos(x9) - cos(x7)*sin(x9))*x5 + (cos(x7)*sin(x8)*cos(x9) + sin(x7)*sin(x9))*x6",
						"cos(x8)*sin(x9)*x4 + (sin(x7)*sin(x8)*sin(x9) + cos(x7)*cos(x9))*x5 + (cos(x7)*sin(x8)*sin(x9) - sin(x7)*cos(x9))*x6",
						"sin(x8)*x4 - sin(x7)*cos(x8)*x5 - cos(x7)*cos(x8)*x6",
						"x12*x5 - x11*x6 - 9.81*sin(x8)",
						"x10*x6 - x12*x4 + 9.81*cos(x8)*sin(x7)",
						"x11*x4 - x10*x5 + 9.81*cos(x8)*cos(x7) - 9.81 - u0 / 1.4",
						"x10 + (sin(x7)*(sin(x8)/cos(x8)))*x11 + (cos(x7)*(sin(x8)/cos(x8)))*x12",
						"cos(x7)*x11 - sin(x7)*x12",
						"(sin(x7)/cos(x8))*x11 + (cos(x7)/cos(x8))*x12",
						"-0.92592592592593*x11*x12 + 18.51851851851852*u1",
						"0.92592592592593*x10*x12 + 18.51851851851852*u2",
						"0",
						"0","0","0","1"};

	// create the neural network object for the controller
	string comb = argv[6];
	string benchmark_name = "nncs_quad_w" + to_string(int(stod(argv[1]) * 1000)) + "_order" + to_string(stoi(argv[4])) + "_" + comb;
	NeuralNetwork nn_controller("quad_controller_3_64");

	// create the NNCS object
	NNCS<Real> system(vars, 0.1, derivatives, nn_controller);


	// Flow* reachability setting
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order);
	setting.setCutoffThreshold(1e-7);
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(vars.size(), I);
	setting.setRemainderEstimation(remainder_estimation);


	// POLAR setting
	unsigned int taylor_order = order;	// same as the flowpipe order, but it is not necessary
	unsigned int bernstein_order = stoi(argv[3]);
	unsigned int partition_num = 4000;
	unsigned int if_symbo = stoi(argv[5]);

	PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Symbolic");
	if(if_symbo == 0){
			// not using symbolic remainder
			polar_setting.set_remainder_type("Concrete");
			polar_setting.symb_rem = if_symbo;
		}

	// initial set
	double w = stod(argv[1]);

	// define the initial set which is a box
	Interval
        init_x1(0 - w, 0 + w),
        init_x2(0 - w, 0 + w),
        init_x3(0 - w, 0 + w),
        init_x4(0 - w, 0 + w),
        init_x5(0 - w, 0 + w),
        init_x6(0 - w, 0 + w),
        init_x7(0),
        init_x8(0),
        init_x9(0),
        init_x10(0),
        init_x11(0),
        init_x12(0);

	// the initial values of the rest of the state variables are 0
	vector<Interval> box(vars.size());
	box[x1_id] = init_x1;
	box[x2_id] = init_x2;
	box[x3_id] = init_x3;
	box[x4_id] = init_x4;
	box[x5_id] = init_x5;
	box[x6_id] = init_x6;
	box[x7_id] = init_x7;
	box[x8_id] = init_x8;
	box[x9_id] = init_x9;
	box[x10_id] = init_x10;
	box[x11_id] = init_x11;
	box[x12_id] = init_x12;

	// translate the initial set to a flowpipe
	Flowpipe initialSet(box);


	// unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// run the reachability computation
	clock_t begin, end;
	double seconds;
	begin = clock();

	double n = stod(argv[2]); // total number of control steps

	Symbolic_Remainder sr(initialSet, 2000);

	system.reach(result, initialSet, n, setting, polar_setting, safeSet, sr);

	// end box or target set
	vector<Interval> end_box;
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
	string reach_result;
	reach_result = "Verification result: Unknown";
	
	// time cost
	end = clock();
	printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
	seconds = (end - begin) / CLOCKS_PER_SEC;
	std::string running_time ="Running Time: %lf\n" + to_string(seconds) + " seconds";

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
	plot_setting.setOutputDims("t", "x3");
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", benchmark_name + "_" + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);
	
	return 0;
}
