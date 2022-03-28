#include "../POLAR/NeuralNetwork.h"
//#include "../NNTaylor.h"
//#include "../domain_computation.h"
//#include "../dynamics_linearization.h"

using namespace std;
using namespace flowstar;

#ifndef PI
#define PI numbers::pi
#endif

#ifndef rejoin_radius
#define rejoin_radius 500
#endif

vector<Interval> copy_domain(const vector<Interval> &domain){
	vector<Interval> new_domain;
	for(Interval itvl: domain){
		Interval new_itvl(itvl);
		new_domain.push_back(new_itvl);
	}
	return new_domain;
}

void build_winman_frame_rot_mat(vector<TaylorModelVec<Real> & winman_fram_rot_mat, const TaylorModelVec<Real> &states, const vector<Real> & domain, const unsigned int order, const Interval & cutoff_threshold, const Global_Setting & setting) {
	// Need to initialize winman_fram_rot_mat from outside
	for(int i = 0; i < 4; i++) winman_fram_rot_mat.tms.push_back(TaylorModel<Real>{});
	states.tms[12].cos_taylor(winman_fram_rot_mat.tms[0], domain, order, cutoff_threshold, setting);
	states.tms[12].sin_taylor(winman_fram_rot_mat.tms[1], domain, order, cutoff_threshold, setting);
	(states.tms[12] * -1.0).sin_taylor(winman_fram_rot_mat.tms[2], domain, order, cutoff_threshold, setting);
	states.tms[12].cos_taylor(winman_fram_rot_mat.tms[3], domain, order, cutoff_threshold, setting);
}

TaylorModelVec<Real> rotate(const TaylorModel<Real> & state1, const TaylorModel<Real> & state2, const vector<TaylorModelVec<Real> & winman_fram_rot_mat, const vector<Real> & domain, const unsigned int order, const Interval & cutoff_threshold, const Global_Setting & setting) {
	TaylorModelVec<Real> tms;
	TaylorModel<Real> tm1;
	state1.mul_ctrunc(tm1, winman_fram_rot_mat[0], domain, order, cutoff_threshold);
	TaylorModel<Real> tm2;
	states2.mul_ctrunc(tm2, winman_fram_rot_mat[1], domain, order, cutoff_threshold);
	tms.push_back(tm1 + tm2);
	// x3 = x2 * frame[2] + x3 * frame[3];
	TaylorModel<Real> tm3;
	state1.mul_ctrunc(tm3, winman_fram_rot_mat[2], domain, order, cutoff_threshold);
	TaylorModel<Real> tm4;
	state2.mul_ctrunc(tm4, winman_fram_rot_mat[3], domain, order, cutoff_threshold);
	tms.push_back(tm3 + tm4);
	return tms;
}

void preprocess(vector<TaylorModelVec<Real> & nn_inputs, vector<TaylorModelVec<Real> & winman_fram_rot_mat, const TaylorModelVec<Real> &states, const TaylorModel<Real> & tm, const vector<Real> & domain, const unsigned int order, const Interval & cutoff_threshold, const Global_Setting & setting) {
	// Need to initialize nn_inputs from outside
	//x1 = x1 / 1000.0
	nn_inputs.push_back(states[0] / 1000.0);
	// x2 = x2 * frame[0] + x3 * frame[1];
	// x3 = x2 * frame[2] + x3 * frame[3];
	TaylorModelVec<Real> tm23 = rotate(states[1], states[2], winman_fram_rot_mat, domain, order, cutoff_threshold);
	//tm23.normalize(domain, cutoff_threshold);
	nn_inputs.insert(end(nn_inputs), begin(tm23), end(tm23));
	// x4 = x4 / 1000.0;
	nn_inputs.push_back(states[3] / 1000.0);
	// x5 = x5 * frame[0] + x6 * frame[1];
	// x6 = x5 * frame[2] + x6 * frame[3];
	TaylorModelVec<Real> tm56 = rotate(states[4], states[5], winman_fram_rot_mat, domain, order, cutoff_threshold);
	//tm56.normalize(domain, cutoff_threshold);
	nn_inputs.insert(end(nn_inputs), begin(tm56), end(tm56));
	// x7 = x7 / 400.0;
	nn_inputs.push_back(states[6] / 400.0);
	// x8 = x8 * frame[0] + x9 * frame[1];
	// x9 = x8 * frame[2] + x9 * frame[3];
	TaylorModelVec<Real> tm89 = rotate(states[7], states[8], winman_fram_rot_mat, domain, order, cutoff_threshold);
	//tm89.normalize(domain, cutoff_threshold);
	nn_inputs.insert(end(nn_inputs), begin(tm89), end(tm89));
    // x10 = x10 / 400.0;
    nn_inputs.push_back(states[9] / 400.0);
	// x11 = x11 * frame[0] + x12 * frame[1];
	// x12 = x11 * frame[2] + x12 * frame[3];
	TaylorModelVec<Real> tm1112 = rotate(states[10], states[11], winman_fram_rot_mat, domain, order, cutoff_threshold);
	//tm1112.normalize(domain, cutoff_threshold);
	nn_inputs.insert(end(nn_inputs), begin(tm1112), end(tm1112));
}


int main(int argc, char *argv[])
{
	intervalNumPrecision = 600;
	
	
	unsigned int numVars = 9;
	Variables vars;
	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x1_id = vars.declareVar("x1"); // rel_dist_lead
	int x2_id = vars.declareVar("x2");	// rel_x_lead
	int x3_id = vars.declareVar("x3");	// rel_y_lead
	int x4_id = vars.declareVar("x4");	// rel_dist_rej
	int x5_id = vars.declareVar("x5"); // rel_x_rej
	int x6_id = vars.declareVar("x6");	// rel_y_rej
	int x7_id = vars.declareVar("x7");	// wvel
	int x8_id = vars.declareVar("x8");	// x_wvel
	int x9_id = vars.declareVar("x9");	// y_wvel
	int x10_id = vars.declareVar("x10");	// lvel
	int x11_id = vars.declareVar("x11");	// x_lvel
	int x12_id = vars.declareVar("x12");	// y_lvel

	int x13_id = vars.declareVar("x13");	// hd_w
	int x14_id = vars.declareVar("x14");	// hd_l

	int u1_id = vars.declareVar("u1");	// u1_lacc
	int u2_id = vars.declareVar("u2");	// u2_lacc	
	int u3_id = vars.declareVar("u3");	// u1_wacc
	int u4_id = vars.declareVar("u4");	// u2_wacc

	int domainDim = numVars + 1;
	 
	
	// Define the continuous dynamics.
    // x(1:14) are the original state variables in the matlab code 

	// x1 = x(1); x2 = x(2) / x(1); x3 = x(3) / x(1); 
    // deriv_x1 = deriv_x(1) = (x(2) * deriv_x(2) + x(3) * deriv_x(3)) / x(1) = (x(2) * (x(11) - x(8)) + x(3) * (x(12) - x(9))) / x(1) = x2 * (x(11) - x(8) + x_3 * (x(12) - x(9));
	Expression<Real> deriv_x1("x2 * (x11 * x10 - x8 * x7) + x3 * (x12 * x10 - x9 * x7)", vars); 
    // deriv_x2 = (deriv_x(2) * x(1) - x(2) * deriv_x1) / (x(1)^2) = deriv_x(2) / x(1) - x2 * deriv_x1 / x(1)
	Expression<Real> deriv_x2("(x11 * x10 - x8 * x7) / x1 - x2 * (x2 * (x11 * x10 - x8 * x7) + x3 * (x12 * x10 - x9 * x7)) / x1", vars); 
	// deriv_x3 = (deriv_x(3) * x(1) - x(3) * deriv_x1) / (x(1)^2) = deriv_x(3) / x(1) - x3 * deriv_x1 / x(1)
    Expression<Real> deriv_x3("(x12 * x10 - x9 * x7) / x1 - x3 * (x2 * (x11 * x10 - x8 * x7) + x3 * (x12 * x10 - x9 * x7)) / x1", vars); 
	// x4 = x(4), x5 =  x(5) / x4; x6 = x(6) / x4; 
    // deriv_x4 = deriv_x(4) = (x(5) * deriv_x(5) + x(6) * deriv_x(6)) / x(4) = x5 * deriv_x(5) + x6 * deriv_x(6)
    Expression<Real> deriv_x4("x5 * (x11 * x10 - x8 * x7 - 500 * sin(60 * pi/ 180 + pi + x14) * u1) + x6 * (x12 * x10 - x9 * x7 + 500 * cos(60 * pi/ 180 + pi + x14) * u1)", vars);
	// deriv_x5 = (deriv_x(5) * x(4) - deriv_x(4) * x(5)) / (x(4) * x(4)) = deriv_x(5) / x4 - deriv_x(4) * x5 / x4 
    Expression<Real> deriv_x5("(x11 * x10 - x8 * x7 - 500 * sin(60 * pi/ 180 + pi + x14) * u1) / x4 - x5 * (x5 * (x11 * x10 - x8 * x7 - 500 * sin(60 * pi/ 180 + pi + x14) * u1) + x6 * (x12 * x10 - x9 * x7 + 500 * cos(60 * pi/ 180 + pi + x14) * u1)) / x4 ", vars); 
	// deriv_x6 = (deriv_x(6) * x(4) - deriv_x(6) * x(5)) / (x(4) * x(4)) = deriv_x(6) / x4 - deriv_x(6) * x5 / x4 
    Expression<Real> deriv_x6("(x12 * x10 - x9 * x7 + 500 * cos(60 * pi/ 180 + pi + x14) * u1) / x4 - x6 * (x5 * (x11 * x10 - x8 * x7 - 500 * sin(60 * pi/ 180 + pi + x14) * u1) + x6 * (x12 * x10 - x9 * x7 + 500 * cos(60 * pi/ 180 + pi + x14) * u1)) / x4", vars); 
	// x7 = x(7); x8 = x(8) / x7; x9 = x(9) / x7; 
    Expression<Real> deriv_x7("x8 * (u4 * cos(x13) - u3 * x7 * sin(x13)) + x9 * (u4 * sin(x13) + u3 * x7 * cos(x13))", vars); 
	Expression<Real> deriv_x8("(u4 * cos(x13) - u3 * x7 * sin(x13)) / x7 - x8 * (x8 * (u4 * cos(x13) - u3 * x7 * sin(x13)) + x9 * (u4 * sin(x13) + u3 * x7 * cos(x13))) / x7", vars);
	Expression<Real> deriv_x9("(u4 * sin(x13) + u3 * x7 * cos(x13)) / x7 - x9 * (x8 * (u4 * cos(x13) - u3 * x7 * sin(x13)) + x9 * (u4 * sin(x13) + u3 * x7 * cos(x13))) / x7", vars); 
	// x10 = x(10); x11 = x(11) / x10; x12 = x(12) / x10; 
    Expression<Real> deriv_x10("x11 * (u2 * cos(x14) - u1 * x9 * sin(x14)) + x12 * (u2 * sin(x14) + u1 * x9 * cos(x14))", vars); 
	Expression<Real> deriv_x11("(u2 * cos(x14) - u1 * x9 * sin(x14)) / x10 - x11 * (x11 * (u2 * cos(x14) - u1 * x9 * sin(x14)) + x12 * (u2 * sin(x14) + u1 * x9 * cos(x14))) / x10", vars);
	Expression<Real> deriv_x12("(u2 * sin(x14) + u1 * x9 * cos(x14)) / x10 - x12 * (x11 * (u2 * cos(x14) - u1 * x9 * sin(x14)) + x12 * (u2 * sin(x14) + u1 * x9 * cos(x14))) / x10", vars); 
	
    Expression<Real> deriv_x13("u3", vars); // deriv_x12 = u0
	Expression<Real> deriv_x14("u1", vars); // deriv_x13 = u2
 
	

	Expression<Real> deriv_u1("0", vars);
	Expression<Real> deriv_u2("0", vars);
	Expression<Real> deriv_u3("0", vars);
	Expression<Real> deriv_u4("0", vars);
 

	// Define the continuous dynamics according to 
	

	vector<Expression<Real>> ode_rhs(numVars);
	ode_rhs[x1_id] = deriv_x0;
	ode_rhs[x2_id] = deriv_x1;
	ode_rhs[x3_id] = deriv_x2;
	ode_rhs[x4_id] = deriv_x3;
	ode_rhs[x5_id] = deriv_x4;
	ode_rhs[x6_id] = deriv_x5;
	ode_rhs[x7_id] = deriv_x6;
	ode_rhs[x8_id] = deriv_x8;
	ode_rhs[x9_id] = deriv_x9;
	ode_rhs[x10_id] = deriv_x10;
	ode_rhs[x11_id] = deriv_x11;
	ode_rhs[x12_id] = deriv_x12;
	ode_rhs[x13_id] = deriv_x13;
	ode_rhs[x14_id] = deriv_x14;

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
		init_x13(0, 2 * PI),
		init_x14(0, 2 * PI),
		init_x1(7500 - 100, 7500 + 100)
		init_x2(-1, 1), 
		init_x3((1. - init_x2 * init_x2).sqrt()), 
		init_x5(init_x1 * init_x2 + (rejoin_angle + 180 + init_x14).cos() * rejoin_radius)), 
		init_x6(init_x1 * init_x3 + (rejoin_angle + 180 + init_x14).sin() * rejoin_radius), 
		init_x4((init_x5 * init_x5 + init_x6 * init_x6).sqrt()), 
		init_x4(300 - 100, 300 + 100), 
		init_x8(init_x13.cos()),
		init_x9(init_x13.sin()),
		init_x10(280 - 100, 280 + 100),
		init_x11(init_x14.cos()),
		init_x12(init_x14.sin()),
	 
		init_u0(0),
		init_u1(0),
		init_u2(0),
		init_u3(0); 
		
	std::vector<Interval> X0;
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
		 
		
		TaylorModelVec<Real> tmv_input;
		for (TaylorModel<Real> tm: initial_set.tmvPre.tms) tmv_input.tms.push_back(tm);
		
		TaylorModelVec<Real> winman_fram_rot_mat;
		build_winman_frame_rot_mat(winman_fram_rot_mat, tmv_input, initial_set.domain, order, cutoff_threshold, setting);
		
        vector<Interval> nn_domain = copy_domain(initial_set.domain);
		TaylorModelVec<Real> nn_input;
		preprocess(nn_input, winman_fram_rot_mat, tmv_input, nn_domain, order, cutoff_threshold, setting);
	

        PolarSetting polar_setting(order, bernstein_order, partition_num, "Taylor", "Concrete");
		TaylorModelVec<Real> nn_output;

		// not using symbolic remainder
		// nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);

        // using symbolic remainder
		nn.get_output_tmv_symbolic(tmv_output, nn_input, nn_domain, polar_setting, setting);
		
		 
		Matrix<Interval> rm1(nn.get_num_of_outputs(), 1);
		tmv_output.Remainder(rm1);
		cout << "Neural network taylor remainder: " << rm1 << endl;

		initial_set.tmvPre.tms[u1_id] = tmv_output.tms[0] * 0.0;
		initial_set.tmvPre.tms[u2_id] = tmv_output.tms[1];
		initial_set.tmvPre.tms[u3_id] = tmv_output.tms[2] * 0.0;
		initial_set.tmvPre.tms[u4_id] = tmv_output.tms[3];
			
		cout << "TM -- Propagation" << endl;

		dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			cout << "Flowpipe taylor remainders: " << endl;
			for (int i = 0; i < 14; i++) {
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
