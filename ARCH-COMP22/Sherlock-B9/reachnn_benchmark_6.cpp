#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	string net_name = "";
	string benchmark_name = "reachnn_benchmark_6_" + net_name;
	// Declaration of the state variables.
	unsigned int numVars = 5;

//	intervalNumPrecision = 800;

	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.

    ODE<Real> dynamics({"x1","- x0 + 0.1 * sin(x2)","x3","u","0"}, vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 7;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.025, order);


	// cutoff threshold
	setting.setCutoffThreshold(1e-6);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.001, 0.001);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
//	double w = 0.1;
	int steps = 20;
//	Interval init_x0(-0.76 - w, -0.76 + w), init_x1(-0.44 - w, -0.44 + w), init_x2(0.52 - w, 0.53 + w), init_x3(-0.29 - w, -0.29 + w), init_u(0); //w=0.01
    Interval init_x0(0.6, 0.7), init_x1(-0.7, -0.6), init_x2(-0.4, -0.3), init_x3(0.5, 0.6), init_u(0);



	// subdividing the initial set to smaller boxes 3 x 3 x 2 x 2
	list<Interval> subdiv_x0;
	init_x0.split(subdiv_x0, 3);

	list<Interval> subdiv_x1;
	init_x1.split(subdiv_x1, 3);

	list<Interval> subdiv_x2;
	init_x2.split(subdiv_x2, 2);

	list<Interval> subdiv_x3;
	init_x3.split(subdiv_x3, 2);

	list<Interval>::iterator iter0 = subdiv_x0.begin();

	vector<Flowpipe> initial_sets;
	vector<vector<Interval> > initial_boxes;

	for(; iter0 != subdiv_x0.end(); ++iter0)
	{
		list<Interval>::iterator iter1 = subdiv_x1.begin();

		for(; iter1 != subdiv_x1.end(); ++iter1)
		{
			list<Interval>::iterator iter2 = subdiv_x2.begin();

			for(; iter2 != subdiv_x2.end(); ++iter2)
			{
				list<Interval>::iterator iter3 = subdiv_x3.begin();

				for(; iter3 != subdiv_x3.end(); ++iter3)
				{
					vector<Interval> box(vars.size());
					box[0] = *iter0;
					box[1] = *iter1;
					box[2] = *iter2;
					box[3] = *iter3;
					box[4] = 0;

					Flowpipe initialSet(box);

					initial_sets.push_back(initialSet);
					initial_boxes.push_back(box);
				}
			}
		}
	}




	// no unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = "controllerTora_POLAR";
//    string nn_name = "nn_6_relu";
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
//	Interval cutoff_threshold(-1e-10, 1e-10);
	unsigned int bernstein_order = 2;
	unsigned int partition_num = 1000;

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

	clock_t begin, end;
	begin = clock();

	for(int m=0; m<initial_sets.size(); ++m)
	{
		Flowpipe initial_set = initial_sets[m];

		Symbolic_Remainder symbolic_remainder(initial_set, 1000);

		for (int iter = 0; iter < steps; ++iter)
		{
			cout << "Step " << iter << " starts.      " << endl;
			//vector<Interval> box;
			//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
			TaylorModelVec<Real> tmv_input;

			tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
			tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);
			tmv_input.tms.push_back(initial_set.tmvPre.tms[2]);
			tmv_input.tms.push_back(initial_set.tmvPre.tms[3]);



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


	//		Matrix<Interval> rm1(1, 1);
	//		tmv_output.Remainder(rm1);
	//		cout << "Neural network taylor remainder: " << rm1 << endl;
	//        tmv_output.tms[0].output(cout, vars);
	        cout << endl;

			initial_set.tmvPre.tms[u_id] = tmv_output.tms[0];

			// Always using symbolic remainder
	        dynamics.reach(result, initial_set, 1, setting, safeSet, symbolic_remainder);

			if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
			{
				initial_set = result.fp_end_of_time;
	//			cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
			}
			else
			{
				printf("Terminated due to too large overestimation.\n");

				cout << initial_boxes[m][0] << "\t" << initial_boxes[m][1] << "\t" << initial_boxes[m][2] << "\t" << initial_boxes[m][3] << endl;

				break;
			}
		}
	}

	end = clock();
	printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);

	string reach_result = "XXXXXXX";
/*
	vector<Constraint> targetSet;
	Constraint c1("x0 - 0.2", vars);		// x0 <= 0.2
	Constraint c2("-x0 - 0.1", vars);		// x0 >= -0.1
	Constraint c3("x1 + 0.6", vars);		// x1 <= -0.6
	Constraint c4("-x1 - 0.9", vars);		// x1 >= -0.9

	targetSet.push_back(c1);
	targetSet.push_back(c2);
	targetSet.push_back(c3);
	targetSet.push_back(c4);

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
*/

	time(&end_timer);
	seconds = difftime(start_timer, end_timer);

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

	std::string running_time = "Running Time: " + to_string(-seconds) + " seconds";

	ofstream result_output("./outputs/" + benchmark_name + "_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "Sherlock_B9_x0_x1", result.tmv_flowpipes, setting);


	plot_setting.setOutputDims("x2", "x3");
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "Sherlock_B9_x2_x3", result.tmv_flowpipes, setting);


	return 0;
}
