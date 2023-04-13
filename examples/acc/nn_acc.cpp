#include "../../POLAR/NeuralNetwork.h"
#include <chrono>
//#include "../NNTaylor.h"
//#include "../domain_computation.h"
//#include "../dynamics_linearization.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	intervalNumPrecision = 100;
	
	// Check 'ARCH-COMP20_Category_Report_Artificial_Intelligence_and_Neural_Network_Control_Systems_-AINNCS-_for_Continuous_and_Hybrid_Systems_Plants.pdf'
	// Declaration of the state variables.
	unsigned int numVars = 10;
	Variables vars;
	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x0_id = vars.declareVar("x0"); // v_set
	int x1_id = vars.declareVar("x1");	// T_gap
	int x2_id = vars.declareVar("x2");	// x_lead
	int x3_id = vars.declareVar("x3");	// x_ego
	int x4_id = vars.declareVar("x4"); // v_lead
	int x5_id = vars.declareVar("x5");	// v_ego
	int x6_id = vars.declareVar("x6");	// gamma_lead
	int x7_id = vars.declareVar("x7");	// gamma_ego
	int t_id = vars.declareVar("t");    // time t

	int u0_id = vars.declareVar("u0");	// a_ego
	
	  int domainDim = numVars + 1;
	 



	// Define the continuous dynamics.
	ODE<Real> dynamics({"0",
						"0",
						"x4",
						"x5",
						"x6",
						"x7",
						"-4 - 2 * x6 - 0.0001 * x4^2",
						"2 * u0 - 2 * x7 - 0.0001 * x5^2",
						"1","0"}, vars);



	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	//setting.setFixedStepsize(0.005, order); // order = 4/5
	setting.setFixedStepsize(0.1, order);

	// cutoff threshold
	setting.setCutoffThreshold(1e-5); //core dumped

	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	double w = stoi(argv[1]); // 0.5
	int steps = stoi(argv[2]);
	Interval init_x0(30), init_x1(1.4), init_x2(90, 91), init_x3(10, 11), init_x4(32, 32.05), init_x5(30, 30.05), init_x6(0), init_x7(0), init_t(0);

	Interval init_u0(0); 
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_x6);
	X0.push_back(init_x7);
	X0.push_back(init_t);
	X0.push_back(init_u0);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	//Symbolic_Remainder symbolic_remainder(initial_set, 500);
	Symbolic_Remainder symbolic_remainder(initial_set, 100);

	// no unsafe set
	vector<Constraint> safeSet, targetSet;

	Constraint c1("x4 - 22.87", vars);		// 
	Constraint c2("-x4 + 22.81 ", vars);	// 
	Constraint c3("x5 - 30.02", vars);		// 
	Constraint c4("-x5 + 29.88", vars);		// 

	targetSet.push_back(c1);
	targetSet.push_back(c2);
	targetSet.push_back(c3);
	targetSet.push_back(c4);

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = "./acc_tanh20x20x20_";
 
	NeuralNetwork nn(nn_name);


	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-12, 1e-12);
	unsigned int bernstein_order = stoi(argv[3]);
	//unsigned int partition_num = 4000;
	unsigned int partition_num = 10;

	unsigned int if_symbo = stoi(argv[5]);
 
	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	vector<string> state_vars;
	state_vars.push_back("x0");
	state_vars.push_back("x1");
	state_vars.push_back("x2");
	state_vars.push_back("x3");
	state_vars.push_back("x4");
	state_vars.push_back("x5");
	state_vars.push_back("x6");
	state_vars.push_back("x7");
	state_vars.push_back("t");
	state_vars.push_back("u0");
	
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
		
		TaylorModelVec<Real> tmv_input;

		for (int i = 0; i < 6; i++)
		{
			tmv_input.tms.push_back(initial_set.tmvPre.tms[i]);
		}

		// taylor propagation (new)
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Taylor", "Concrete");
		polar_setting.set_num_threads(-1);
		TaylorModelVec<Real> tmv_output;
		//tmv_output.tms.push_back(TaylorModel<Real>(-1.7690021, 1));

		nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);

		
		Matrix<Interval> rm1(nn.get_num_of_outputs(), 1);
		tmv_output.Remainder(rm1);
		cout << "Neural network taylor remainder: " << rm1 << endl;


		initial_set.tmvPre.tms[u0_id] = tmv_output.tms[0];
			
		cout << "TM -- Propagation" << endl;

		dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);


		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			cout << "Flowpipe taylor remainders: " << endl;
			for (int i = 0; i < 6; i++) {
				cout << initial_set.tmv.tms[i].remainder << endl;
			}
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			return 1;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
	seconds = elapsed.count() *  1e-9;
	printf("Time measured: %.3f seconds.\n", seconds);

	vector<Interval> end_box;
	string reach_result;
	// reach_result = "Verification result: Unknown(35)";
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);

	bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
	if(b)
	{
		reach_result = "Verification result: Yes(" + to_string(steps) + ")";
	}
	else
	{
		reach_result = "Verification result: No(" + to_string(steps) + ")";
	}

	// time(&end_timer);
	// seconds = difftime(start_timer, end_timer);
	// printf("time cost: %lf\n", -seconds);

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);
	plot_setting.setOutputDims("x4", "x5");

	int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	std::string running_time = "Running Time: " + to_string(seconds) + " seconds";

	ofstream result_output("./outputs/acc_tanh20x20x20_x4x5_steps_" + to_string(steps) + "_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	cout << reach_result << endl;
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs


	plot_setting.plot_2D_octagon_MATLAB("./outputs/", "acc_"  + to_string(steps) + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);
 

	return 0;
}
