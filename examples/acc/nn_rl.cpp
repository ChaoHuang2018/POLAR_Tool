#include "../POLAR/NeuralNetwork.h"
//#include "../NNTaylor.h"
//#include "../domain_computation.h"
//#include "../dynamics_linearization.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	intervalNumPrecision = 600;
	
	/*
	// Declaration of the state variables.
	unsigned int numVars = 7;
	
	// intput format (\omega_1, \psis_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x0_id = stateVars.declareVar("x0"); // x_lead
	int x1_id = stateVars.declareVar("x1");	// v_lead
	int x2_id = stateVars.declareVar("x2");	// gamma_lead
	int x3_id = stateVars.declareVar("x3");	// x_ego
	int x4_id = stateVars.declareVar("x4");	// v_ego
	int x5_id = stateVars.declareVar("x5"); // gamma_ego
	int u0_id = stateVars.declareVar("u0");
	 
	int domainDim = numVars + 1;
	 
	// Define the continuous dynamics.
	Expression<Real> deriv_x0("x1"); // theta_r = 0
	Expression<Real> deriv_x1("x2"); 
	Expression<Real> deriv_x2("-2 * 2 - 2 * x2 - 0.0001 * x1^2 ");
	Expression<Real> deriv_x2("x3");
	Expression<Real> deriv_x3("x4");
	Expression<Real> deriv_x4("2 * u0 - 2 * x4 - 0.0001 * x3^2");
	Expression<Real> deriv_u0("0");
	 
	// Define the continuous dynamics according to 
	

	vector<Expression<Real>> ode_rhs(numVars);
	ode_rhs[x0_id] = deriv_x0;
	ode_rhs[x1_id] = deriv_x1;
	ode_rhs[x2_id] = deriv_x2;
	ode_rhs[x3_id] = deriv_x3;
	ode_rhs[x4_id] = deriv_x4;
	ode_rhs[x5_id] = deriv_x5;
	ode_rhs[u0_id] = deriv_u0;
	*/
	
	// Check 'ARCH-COMP20_Category_Report_Artificial_Intelligence_and_Neural_Network_Control_Systems_-AINNCS-_for_Continuous_and_Hybrid_Systems_Plants.pdf'
	// Declaration of the state variables.
	unsigned int numVars = 9;
	Variables vars;
	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x0_id = vars.declareVar("x0"); // rel_dist
	int x1_id = vars.declareVar("x1");	// x_ref
	int x2_id = vars.declareVar("x2");	// y_ref
	int x3_id = vars.declareVar("x3");	// rej_dist
	int x4_id = vars.declareVar("x4"); // x_rej
	int x5_id = vars.declareVar("x5");	// y_rej
	int x6_id = vars.declareVar("x6");	// wvel
	int x7_id = vars.declareVar("x7");	// x_wvel
	int x8_id = vars.declareVar("x8");	// y_wvel
	int x9_id = vars.declareVar("x9");	// lvel
	int x10_id = vars.declareVar("x10");	// x_lvel
	int x11_id = vars.declareVar("x11");	// y_lvel

	int x12_id = vars.declareVar("x12");	// hd_l
	int x13_id = vars.declareVar("x13");	// hd_w

	// normalization
	int x14_id = vars.declareVar("x14");	// hd_l
	int x15_id = vars.declareVar("x15");	// hd_w
	int x16_id = vars.declareVar("x16");	// hd_l
	int x17_id = vars.declareVar("x17");	// hd_w

	int u0_id = vars.declareVar("u0");	// u1_lacc
	int u1_id = vars.declareVar("u1");	// u2_lacc	
	int u2_id = vars.declareVar("u2");	// u1_wacc
	int u3_id = vars.declareVar("u3");	// u2_wacc

	int domainDim = numVars + 1;
	 
	// Define the continuous dynamics.
	Expression<Real> deriv_x0("(x1 * (x10 - x7) + x2 * (x11 - x8)) / x0", vars); //  deriv_x0 = (2 * x1 * deriv(x1) + 2 * x2 * deriv(x2))/2sqrt(x1*x1 + x2 * x2)
	Expression<Real> deriv_x1("x10 - x7", vars); //  deriv_x1 = x10 - x7
	Expression<Real> deriv_x2("x11 - x8", vars); // deriv_x2 = x11 - x8
	Expression<Real> deriv_x3("((x4 * (x10 - x8 - 500 * sin(60 * pi/ 180 + 2 * pi + x12)) * u0 * 180 / pi + x5 * (x11 - x8 + 500 * cos(60 * pi/ 180 + 2 * pi + x12)) * u0 * 180 / pi)) / x3", vars); // deriv_x3 = (2 * x4 * deriv(x4) + 2 * x5 * deriv(x5))/sqrt(x4*x4 + x5 * x5)
	Expression<Real> deriv_x4("x10 - x7 - 500 * sin(60 * pi / 180 + pi + x12) * u0 * 180 / pi", vars); // deriv_x4 = x7
	Expression<Real> deriv_x5("x11 - x8 + 500 * cos(60 * pi / 180 + pi + x12) * u0 * 180 / pi", vars); // deriv_x5 = x8
	Expression<Real> deriv_x6("u3", vars); // deriv_x6 = (2 * x7 * deriv(x7) + 2 * x8 * deriv(x8))/sqrt(x7*x7 + x8 * x8)
	Expression<Real> deriv_x7("u3 * cos(x12 * pi / 180) - u2 * 180 / pi * x6 * sin(x12 * pi / 180)", vars); // deriv_x7 = u3 * x7/x6 - u2 * x8
	Expression<Real> deriv_x8("u3 * sin(x12 * pi / 180) + u2 * 180 / pi * x6 * cos(x12 * pi / 180)", vars); // deriv_x8 = u3 * x8/x6 + u2 * x7
	Expression<Real> deriv_x9("u1", vars); // deriv_x9 = (2 * x10 * deriv(x10) + 2 * x11 * deriv(x11))/sqrt(x10*x10 + x11 * x11)
	Expression<Real> deriv_x10("u1 * cos(x13 * pi / 180) - u0 * 180 / pi * x9 * sin(x13 * pi / 180)", vars); // deriv_x10 = u1 * x11/x10 - u1 * x12
	Expression<Real> deriv_x11("u1 * sin(x13 * pi / 180) + u0 * 180 / pi * x9 * cos(x13 * pi / 180)", vars); // deriv_x11 = u1 * x12/x10 - u1 * x11
	Expression<Real> deriv_x12("u0 * 180 / pi", vars); // deriv_x12 = u0
	Expression<Real> deriv_x13("u2 * 180 / pi", vars); // deriv_x13 = u2

	Expression<Real> deriv_x14("(x1 * (x10 - x7) + x2 * (x11 - x8)) / (1000.0 * x0)", vars); //  deriv_x0 = (2 * x1 * deriv(x1) + 2 * x2 * deriv(x2))/2sqrt(x1*x1 + x2 * x2)
	Expression<Real> deriv_x15("((x4 * (x10 - x8 - 500 * sin(60 * pi/ 180 + 2 * pi + x12)) * u0 * 180 / pi + x5 * (x11 - x8 + 500 * cos(60 * pi/ 180 + 2 * pi + x12)) * u0 * 180 / pi)) / (1000.0 * x3)", vars); // deriv_x3 = (2 * x4 * deriv(x4) + 2 * x5 * deriv(x5))/sqrt(x4*x4 + x5 * x5)
	Expression<Real> deriv_x16("u3 / 400.0", vars); // deriv_x6 = (2 * x7 * deriv(x7) + 2 * x8 * deriv(x8))/sqrt(x7*x7 + x8 * x8)
	Expression<Real> deriv_x17("u1 / 400.0", vars); // deriv_x9 = (2 * x10 * deriv(x10) + 2 * x11 * deriv(x11))/sqrt(x10*x10 + x11 * x11)
	

	// Define the continuous dynamics according to 
	

	vector<Expression<Real>> ode_rhs(numVars);
	ode_rhs[x0_id] = deriv_x0;
	ode_rhs[x1_id] = deriv_x1;
	ode_rhs[x2_id] = deriv_x2;
	ode_rhs[x3_id] = deriv_x3;
	ode_rhs[x4_id] = deriv_x4;
	ode_rhs[x5_id] = deriv_x5;
	ode_rhs[x6_id] = deriv_x6;
	ode_rhs[x8_id] = deriv_x8;
	ode_rhs[x9_id] = deriv_x9;
	ode_rhs[x10_id] = deriv_x10;
	ode_rhs[x11_id] = deriv_x11;
	ode_rhs[x12_id] = deriv_x12;
	ode_rhs[x13_id] = deriv_x13;
	
	ode_rhs[x14_id] = deriv_x14;
	ode_rhs[x15_id] = deriv_x15;
	ode_rhs[x16_id] = deriv_x16;
	ode_rhs[x17_id] = deriv_x17;

	ode_rhs[u0_id] = deriv_u0;
	ode_rhs[u1_id] = deriv_u1;
	ode_rhs[u2_id] = deriv_u2;
	ode_rhs[u3_id] = deriv_u3;

	Deterministic_Continuous_Dynamics dynamics(ode_rhs);

	// Specify the parameters for reachability computation.
	Computational_Setting setting;

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order); // order = 4/5
	//setting.setFixedStepsize(0.005, order); // order = 4/5

	// time horizon for a single control step
	setting.setTime(0.1);

	// cutoff threshold
	setting.setCutoffThreshold(1e-10); //core dumped
	//setting.setCutoffThreshold(1e-7);

	// queue size for the symbolic remainder
	// Not in the newest version??
	//setting.setQueueSize(2000); //200
	// symbolic remainder object (, 2000)

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
	double w = stod(argv[1]); // 0.5
	int steps = stoi(argv[2]);
	Interval 
		init_x0(30), 
		init_x1(1.4), 
		init_x2(90, 110), 
		init_x3(10, 11), 
		init_x4(32, 32.2), 
		init_x5(30, 30.2), 
		init_x6(0), 
		init_x7(0),
		init_x8(0),
		init_x6(0),
		init_x9(250, 300),
		init_x10(0),
		init_x11(0),
		init_x12(0),
		init_x13(0);

		init_x14(30/1000);
		init_x15(10/1000.0, 11/1000.0);
		init_x16(0);
		init_x17(0);

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
	X0.push_back(init_x8);
	X0.push_back(init_x9);
	X0.push_back(init_x10);
	X0.push_back(init_x11);
	X0.push_back(init_x12);
	X0.push_back(init_x13);
	X0.push_back(init_x14);
	X0.push_back(init_x15);
	X0.push_back(init_x16);
	X0.push_back(init_x17);
	X0.push_back(init_u0);
	X0.push_back(init_u1);
	X0.push_back(init_u2);
	X0.push_back(init_u3);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);
	Symbolic_Remainder symbolic_remainder(initial_set, 500);
	// no unsafe set
	vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = "../networks/acc/acc_tanh20x20x20_";
 
	NeuralNetwork nn(nn_name);

	/* Not in the newest version???
	unsigned int maxOrder = 15;
	Global_Computation_Setting g_setting;
	g_setting.prepareForReachability(maxOrder);
	*/

	// the order in use
	// unsigned int order = 5;
	//Interval cutoff_threshold(-1e-7, 1e-7);
	Interval cutoff_threshold(-1e-10, 1e-10);
	unsigned int bernstein_order = stoi(argv[3]);
	unsigned int partition_num = 4000;

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
	state_vars.push_back("x8");
	state_vars.push_back("x9");
	state_vars.push_back("x10");
	state_vars.push_back("x11");
	state_vars.push_back("x12");
	state_vars.push_back("x13");
	state_vars.push_back("x14");
	state_vars.push_back("x15");
	state_vars.push_back("x16");
	state_vars.push_back("x17");
	
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
		//TaylorModelVec<Real> tmv_temp;
		//initial_set.compose(tmv_temp, order, cutoff_threshold);
		/*
		
		*/
		for (int i = 0; i < new int[]{14, 1, 2, 15, 4, 5, 16, 7, 8, 17, 10, 11}; i++)
		{
			tmv_input.tms.push_back(initial_set.tmvPre.tms[i]);
			//tmv_input.tms.push_back(tmv_temp.tms[i]);
	 
		}

		// taylor propagation (old)
		/*
		NNTaylor nn_taylor(nn);
		TaylorInfo ti(g_setting, order, bernstein_order, partition_num, cutoff_threshold);
		TaylorModelVec<Real> tmv_output;
		if (if_symbo == 0)
		{
			nn_taylor.get_output_tmv(tmv_output, tmv_input, ti, initial_set.domain);
		}
		else
		{
			nn_taylor.NN_Reach(tmv_output, tmv_input, ti, initial_set.domain);
		}
		// cout << "initial_set.domain: " << initial_set.domain[0] << initial_set.domain[1] << endl;
		*/

		// taylor propagation (new)
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Taylor", "Concrete");
		TaylorModelVec<Real> tmv_output;

		// not using symbolic remainder
		// nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);

        // using symbolic remainder
		nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);

		
		Matrix<Interval> rm1(nn.get_num_of_outputs(), 1);
		tmv_output.Remainder(rm1);
		cout << "Neural network taylor remainder: " << rm1 << endl;

		// taylor
		// NNTaylor nn_taylor1(nn);
		// vector<Interval> box;
		// initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		// cout << "initial_set: " << box[0] << box[1] << endl;
		// vector<Interval> box_state;
		// for (int i = 0; i < state_vars.size(); i++)
		// {
		// 	box_state.push_back(box[i]);
		// }
		// nn_taylor1.set_taylor_linear(state_vars, box_state);
		// cout << "11111" << endl;
		// vector<double> jcb = nn_taylor1.get_jacobian();
		// vector<Real> jcb_real;
		// for (int i = 0; i < jcb.size(); i++)
		// {
		// 	jcb_real.push_back(Real(jcb[i]));
		// }
		// Interval rem = nn_taylor1.get_taylor_remainder() + nn_taylor1.get_output();
		// cout << "Whole remainder: " << rem << endl;
		// Polynomial<Real> tm_coef(jcb_real);
		// TaylorModel<Real> tm_output2(jcb_real, rem);

		// if (rm1[0][0].width() < rem.width())
		 
		initial_set.tmvPre.tms[u0_id] = tmv_output.tms[0];
			
		cout << "TM -- Propagation" << endl;

		dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);

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

	vector<Interval> end_box;
	string reach_result;
	reach_result = "Verification result: Unknown(35)";
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);

	time(&end_timer);
	seconds = difftime(start_timer, end_timer);

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

	std::string running_time = "Running Time: " + to_string(-seconds) + " seconds";

	ofstream result_output("./outputs/rl_tanh256x256_" + to_string(steps) + "_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", "rl_tanh256x256_"  + to_string(steps) + "_"  + to_string(if_symbo), result);

 

	return 0;
}
