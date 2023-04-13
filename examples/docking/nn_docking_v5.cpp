#include "../../POLAR/NeuralNetwork.h"
//#include "../NNTaylor.h"
//#include "../domain_computation.h"
//#include "../dynamics_linearization.h"
#include <math.h>
using namespace std;
using namespace flowstar;

#ifndef PI
#define PI atan(1.0)*4
#endif

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
	unsigned int numVars = 10;
	Variables vars;
	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x1_id = vars.declareVar("x1"); // x_pos
	int x2_id = vars.declareVar("x2");	// y_pos
	int x3_id = vars.declareVar("x3");	// x_vel
	int x4_id = vars.declareVar("x4");	// y_vel
	int x5_id = vars.declareVar("x5"); // ||v||
	int x6_id = vars.declareVar("x6");	// max_vel
	//int x7_id = vars.declareVar("x7");	// norm_x_pos
	//int x8_id = vars.declareVar("x8");	// norm_y_pos
	//int x9_id = vars.declareVar("x9");	// norm_x_vel
	//int x10_id = vars.declareVar("x10");	// norm_y_vel
 

	int u1_id = vars.declareVar("u1");	// fx_mean
	int u2_id = vars.declareVar("u2");	// fx_std	
	int u3_id = vars.declareVar("u3");	// fy_mean
	int u4_id = vars.declareVar("u4");	// fy_std
 
	 
	// Define the continuous dynamics.
	Expression<Real> deriv_x1("x3", vars); //  
	Expression<Real> deriv_x2("x4", vars); //   
	Expression<Real> deriv_x3("2.0 * 0.001027 * x4 + 3 * 0.001027 * 0.001027 * x1 + u1 / 12.", vars); //  
	Expression<Real> deriv_x4("-2.0 * 0.001027 * x3 + u3 / 12.", vars); //  
	Expression<Real> deriv_x5("((2.0 * 0.001027 * x4 + 3 * 0.001027 * 0.001027 * x1 + u1 / 12.) * x3 + (-2.0 * 0.001027 * x3 + u3 / 12.) * x4) / x5", vars);  
	Expression<Real> deriv_x6("2.0 * 0.001027 * (x1 * x3 + x2 * x4) / sqrt(x1 * x1 + x2 * x2)", vars);  
	//Expression<Real> deriv_x7("x2 / 1000.0", vars);  
	//Expression<Real> deriv_x8("x3 / 1000.0", vars); // deriv_x7 = u3 * x7/x6 - u3 * x8
	//Expression<Real> deriv_x9("(2.0 * 0.001027 * x3 + 3 * 0.001027 * 0.001027 * x1 + u0 / 12.) / 0.5", vars); // deriv_x8 = u3 * x8/x6 + u3 * x7
	//Expression<Real> deriv_x10("(-2.0 * 0.001027 * x2 + u1 / 12.) / 0.5", vars); // deriv_x9 = (2 * x10 * deriv(x10) + 2 * x11 * deriv(x11))/sqrt(x10*x10 + x11 * x11)
	  
	// Define the continuous dynamics according to 
	Expression<Real> deriv_u1("0", vars);
	Expression<Real> deriv_u2("0", vars);
	Expression<Real> deriv_u3("0", vars);
	Expression<Real> deriv_u4("0", vars);
	
	vector<Expression<Real>> ode_rhs(numVars);
	 
	ode_rhs[x1_id] = deriv_x1;
	ode_rhs[x2_id] = deriv_x2;
	ode_rhs[x3_id] = deriv_x3;
	ode_rhs[x4_id] = deriv_x4;
	ode_rhs[x5_id] = deriv_x5;
	ode_rhs[x6_id] = deriv_x6;
	//ode_rhs[x8_id] = deriv_x8;
	//ode_rhs[x9_id] = deriv_x9;

	 
	ode_rhs[u1_id] = deriv_u1;
	ode_rhs[u2_id] = deriv_u2;
	ode_rhs[u3_id] = deriv_u3;
	ode_rhs[u4_id] = deriv_u4;

	Deterministic_Continuous_Dynamics dynamics(ode_rhs);

	// Specify the parameters for reachability computation.
	Computational_Setting setting;

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order); // order = 4/5
	//setting.setFixedStepsize(0.005, order); // order = 4/5

	// time horizon for a single control step
	setting.setTime(1);

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
	int w = stod(argv[1]); // 0.5
 
	int steps = stoi(argv[2]);

	Interval 
		//init_x1(125 - 25, 125 + 25), 
		//init_x2(125 - 25, 125 + 25); 
		init_x1(25 - 1, 25 + 1), 
		init_x2(25 - 1, 25 + 1); 
	
	Interval 
		init_x6(init_x1.pow(2) + init_x2.pow(2));
		init_x6.sqrt_assign();
		init_x6.mul_assign(2.0 * 0.001027);
		init_x6.add_assign(0.2);
	cout << "1111" << endl;
	
	 
	 
	Interval
		//init_x3(init_x5 * Interval(-1, 1));
		//init_x3(init_x6 * (-0.7));
		init_x3(-init_x6.sup() * 0.5, -init_x6.sup() * 0.5);
	Interval 
		//init_x5(init_x6 * Interval(0, 1));
		//init_x4(init_x6 * (-0.7));
		init_x4(-init_x6.sup() * 0.5, -init_x6.sup() * 0.5);

	Interval
		init_x5(init_x3.pow(2) + init_x4.pow(2));
		init_x5.abs_assign();
		init_x5.sqrt_assign();	
	
	Interval 
		init_u1(0),
		init_u2(0),
		init_u3(0),
		init_u4(0); 

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
	X0.push_back(init_u4);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);
	Symbolic_Remainder symbolic_remainder(initial_set, 500);
	// no unsafe set
	vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = "docking_tanh" + to_string(w) + "x" + to_string(w) + "_tanh";
 
	NeuralNetwork nn(nn_name);
	
	const string dir_name =  "./outputs/docking_v5_" + nn_name;
	char* c = const_cast<char*>(dir_name.c_str());
		cout << c << endl;
	int mkres = mkdir(c, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}
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
		
 

		TaylorModelVec<Real> tmv_temp;
		initial_set.compose(tmv_temp, order, cutoff_threshold);
		
		TaylorModelVec<Real> tmv_input;
		//TaylorModelVec<Real> tmv_temp;
		//initial_set.compose(tmv_temp, order, cutoff_threshold);
		/*
		
		*/
		for (int i = 0; i < 6; i++)
		{
			double norm = 1.;
			if(i <= 1) norm = 1000.0;
			else if(i <= 3) norm = 0.5;
			else norm = 1.;
			tmv_input.tms.push_back(initial_set.tmvPre.tms[i] / norm);
			//tmv_input.tms.push_back(tmv_temp.tms[i]);
	 
		}

		for (TaylorModel<Real> tm: tmv_input.tms) {
			Interval box;
			tm.intEval(box, initial_set.domain);
			cout << "Initial nn input interval: [" << box.inf() << ", " << box.sup() << "]" << endl; 
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
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
		TaylorModelVec<Real> tmv_output;

		// not using symbolic remainder
		// nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);

        // using symbolic remainder
		nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);

		
		Matrix<Interval> rm1(nn.get_num_of_outputs(), 1);
		tmv_output.Remainder(rm1);
		cout << "Neural network taylor remainder: " << rm1 << endl;


		initial_set.tmvPre.tms[u1_id] = tmv_output.tms[0];
		initial_set.tmvPre.tms[u2_id] = tmv_output.tms[1] * 0.0;
		initial_set.tmvPre.tms[u3_id] = tmv_output.tms[2];
		initial_set.tmvPre.tms[u4_id] = tmv_output.tms[3] * 0.0;

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
		 
		initial_set.tmvPre.tms[u1_id] = tmv_output.tms[0];
			
		cout << "TM -- Propagation" << endl;

		dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			cout << "Flowpipe taylor remainders: " << endl;
			for (int i = 0; i < numVars; i++) {
				cout << initial_set.tmv.tms[i].remainder << endl;
			}
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			//return 1;
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
	
	

	std::string running_time = "Running Time: " + to_string(-seconds) + " seconds";

	ofstream result_output(dir_name + "/Steps_" + to_string(steps) + "_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
	//plot_setting.setOutputDims("x1", "x2");
	//plot_setting.plot_2D_octagon_MATLAB("./outputs/docking_v5_", nn_name + "_Steps" + to_string(steps) + "_"  + to_string(if_symbo), result);
	plot_setting.setOutputDims("x1", "x2");
	plot_setting.plot_2D_octagon_MATLAB(c, "Steps" + to_string(steps) + "_"  + to_string(if_symbo), result);
	plot_setting.plot_2D_octagon_MATLAB("./outputs/docking_v5_docking_tanh64x64_tanh/", "Steps" + to_string(steps) + "_x1x2_"  + to_string(if_symbo), result);
	plot_setting.plot_2D_octagon_MATLAB("./outputs/docking_v5_docking_tanh64x64_tanh/", "Steps" + to_string(steps) + "_x1x2_"  + to_string(if_symbo), result);
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/docking_v5_docking_tanh64x64_tanh/", "Steps" + to_string(steps) + "_x1x2_"  + to_string(if_symbo), result);
	
 

	return 0;
}
