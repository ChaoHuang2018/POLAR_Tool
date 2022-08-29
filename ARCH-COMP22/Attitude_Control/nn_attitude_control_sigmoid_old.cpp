#include "../../POLAR/NeuralNetwork.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
    
//    intervalNumPrecision = 300;
    
	// Declaration of the state variables.
	unsigned int numVars = 9;
    
    Variables vars;

	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int x5_id = vars.declareVar("x5");
	int u0_id = vars.declareVar("u0");
	int u1_id = vars.declareVar("u1");
	int u2_id = vars.declareVar("u2");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"0.25*(u0 + x1*x2)", "0.5*(u1 - 3*x0*x2)", "u2 + 2*x0*x1", "0.5*x1*(x3^2 + x4^2 + x5^2 - x5) + 0.5*x2*(x3^2 + x4^2 + x4 + x5^2) + 0.5*x0*(x3^2 + x4^2 + x5^2 + 1)", "0.5*x0*(x3^2 + x4^2 + x5^2 + x5) + 0.5*x2*(x3^2 - x3 + x4^2 + x5^2) + 0.5*x1*(x3^2 + x4^2 + x5^2 + 1)", "0.5*x0*(x3^2 + x4^2 - x4 + x5^2) + 0.5*x1*(x3^2 + x3 + x4^2 + x5^2) + 0.5*x2*(x3^2 + x4^2 + x5^2 + 1)"}, vars);
    
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 4;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order);

	// time horizon for a single control step
//	setting.setTime(0.1);

	// cutoff threshold
	setting.setCutoffThreshold(1e-7);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

    //setting.g_setting.prepareForReachability(15);
  //  cout << "--------" << setting.g_setting.factorial_rec.size() << ", " << setting.g_setting.power_4.size() << ", " << setting.g_setting.double_factorial.size() << endl;

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	double w = 0.05;
	int steps = 30;
	Interval init_x0(-0.445 - w, -0.445 + w), init_x1(-0.545 - w, -0.545 + w), init_x2(0.655 - w, 0.655 + w), init_x3(-0.745 - w, -0.745 + w), init_x4(0.855 - w, 0.855 + w), init_x5(-0.645 - w, -0.645 + w);
	// Interval init_x0(-0.25 - w, -0.25 + w), init_x1(-0.25 - w, -0.25 + w), init_x2(0.35 - w, 0.35 + w), init_x3(-0.35 - w, -0.35 + w), init_x4(0.45 - w, 0.45 + w), init_x5(-0.35 - w, -0.35 + w);
	Interval init_u0(0), init_u1(0), init_u2(0);
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_u0);
	X0.push_back(init_u1);
	X0.push_back(init_u2);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);
    
    Symbolic_Remainder symbolic_remainder(initial_set, 2000);

	// no unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
    
	string nn_name = "CLF_controller_layer_num_3";
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-7, 1e-7);
	unsigned int bernstein_order = order;
	unsigned int partition_num = 4000;

	unsigned int if_symbo = 1;

	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	if (if_symbo == 0)
	{
		cout << "High order abstraction starts." << endl;
	}
	else
	{
		cout << "High order abstraction with symbolic remainder starts." << endl;
	}

	// perform 35 control steps
	for (int iter = 0; iter < steps; ++iter)
	{
		cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		TaylorModelVec<Real> tmv_input;

		for (int i = 0; i < 6; i++)
		{
			tmv_input.tms.push_back(initial_set.tmvPre.tms[i]);
		}


		// taylor propagation
        // taylor propagation
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
        TaylorModelVec<Real> tmv_output;

        if (if_symbo == 0)
        {
            // not using symbolic remainder
            nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
        }
        else
        {
            // using symbolic remainder
            nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
        }

        // tmv_output.output(cout, vars);
//		Matrix<Interval> rm1(nn.get_num_of_outputs(), 1);
//		tmv_output.Remainder(rm1);
//		cout << "Neural network taylor remainder: " << rm1 << endl;


        initial_set.tmvPre.tms[u0_id] = tmv_output.tms[0];
        initial_set.tmvPre.tms[u1_id] = tmv_output.tms[1];
        initial_set.tmvPre.tms[u2_id] = tmv_output.tms[2];
//        cout << "TM -- Propagation" << endl;

        dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			break;
		}
	}

	vector<Interval> end_box;
	string reach_result;
	reach_result = "Verification result: Unknown(35)";
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);

	time(&end_timer);
	seconds = difftime(start_timer, end_timer);

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);

	int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	std::string running_time = "Running Time: " + to_string(-seconds) + " seconds";

	ofstream result_output("./outputs/nn_ac_sigmoid_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
    plot_setting.setOutputDims("x0", "x1");
    plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "nn_ac_sigmoid_x0_x1_" + to_string(if_symbo), result.tmv_flowpipes, setting);

	plot_setting.setOutputDims("x2", "x3");
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "nn_ac_sigmoid_x2_x3_" + to_string(if_symbo), result.tmv_flowpipes, setting);

	plot_setting.setOutputDims("x4", "x5");
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "nn_ac_sigmoid_x4_x5_" + to_string(if_symbo), result.tmv_flowpipes, setting);

	return 0;
}
