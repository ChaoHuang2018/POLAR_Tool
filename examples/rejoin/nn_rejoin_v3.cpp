#include "../../POLAR/NeuralNetwork.h"
#include "../../flowstar/settings.h"
//#include "../NNTaylor.h"
//#include "../domain_computation.h"
//#include "../dynamics_linearization.h"

using namespace std;
using namespace flowstar;

#ifndef PI
#define PI atan(1.0)*4
#endif

#ifndef rejoin_radius
#define rejoin_radius 500
#endif

#ifndef rejoin_angle
#define rejoin_angle PI * 60.0 / 180.0
#endif


/*
void initialize_states(const TaylorModelVec<Real> & tmv_input, const vector<Real> & domain, const unsigned int order, const Interval & cutoff_threshold, const Global_Setting & setting) {
	
	TaylorModelVec<Real> tmv_temp;
	for(int i = 0; i < 12; i++) tmv_temp.push_back(TaylorModel<Real>());
	// x1 = sqrt(x2 * x2 + x3 * x3)
	tmv_temp.tms[1].mul_ctrunc(tmv_temp.tms[1], tmv_input.tms[1], domain, order, cutoff_threshold);
	tmv_temp.tms[2].mul_ctrunc(tmv_temp.tms[2], tmv_input.tms[2], domain, order, cutoff_threshould);
	(tmv_temp.tms[1] + tmv_temp.tms[2]).sqrt_taylor(tmv_input.tms[0], domain, order, cutoff_threshold, setting);
	// x2 = x2
	// x3 = x3
	// x5 = x1 + rejoin_radius* cosd(numbers::pi + 180 + x13);
	tmv_input.tms[4] = tmv_input.tms[2] + 500 * (tmv_input[12] + PI + PI / 3).cos_taylor(tmv_temp.tms[4],  domain, order, cutoff_threshold, setting);
	// x6 = x2 + rejoin_radius* sind(numbers::pi + 180 + x13);
	tmv_input.tms[5] = tmv_input.tms[3] + 500 * (tmv_input[12] + PI + PI / 3).sin_taylor(tmv_temp.tms[5],  domain, order, cutoff_threshold, setting);
	// x4 = sqrt(x2 * x2 + x3 * x3)_
	tmv_temp.tms[4] = tmv_input.tms[4];
	tmv_temp.tms[4].mul_ctrunc(tmv_temp.tms[4], tmv_input.tms[4], domain, order, cutoff_threshold);
	tmv_temp.tms[5] = tmv_input.tms[5];
	tmv_temp.tms[5].mul_ctrunc(tmv_temp.tms[5], tmv_input.tms[5], domain, order, cutoff_threshold);
	(tmv_temp.tms[5] + tmv_temp.tms[4]).sqrt_taylor(tmv_input.tms[3], domain, order, cutoff_threshold, setting);
	//x7 = x7
	//x8 = x7 * cosd(x12)
	tmv_input.tms[7] = tmps_input.tms[6];
	tmv_input.tms[7].cos_taylor(tmv_input.tms[11], domain, order, cutoff_threshold, setting);
	//x9 = x7 * sind(x12)
	tmv_input.tms[8] = tmps_input.tms[6];
	tmv_input.tms[8].sin_taylor(tmv_input.tms[11], domain, order, cutoff_threshold, setting);
	//x10 = x10
	//x11 = x10 * cosd(x13)
	tmv_input.tms[10] = tmps_input.tms[9];
	tmv_input.tms[10].cos_taylor(tmv_input.tms[12], domain, order, cutoff_threshold, setting);
	//x12 = x10 * sind(x13)
	tmv_input.tms[11] = tmps_input.tms[9];
	tmv_input.tms[11].sin_taylor(tmv_input.tms[12], domain, order, cutoff_threshold, setting);

}
*/

vector<Interval> copy_domain(const vector<Interval> &domain){
	vector<Interval> new_domain;
	for(Interval itvl: domain){
		Interval new_itvl(itvl);
		new_domain.push_back(new_itvl);
	}
	return new_domain;
}


void build_winman_frame_rot_mat(TaylorModelVec<Real> & winman_fram_rot_mat, const TaylorModel<Real> &state, vector<Interval> & domain, const unsigned int order, const Interval & cutoff_threshold, const Global_Setting & setting) {
	// Need to initialize winman_fram_rot_mat from outside
	for(int i = 0; i < 4; i++) winman_fram_rot_mat.tms.push_back(TaylorModel<Real>{});
	state.cos_taylor(winman_fram_rot_mat.tms[0], domain, order, cutoff_threshold, setting);
	state.sin_taylor(winman_fram_rot_mat.tms[1], domain, order, cutoff_threshold, setting);
	(state * -1.0).sin_taylor(winman_fram_rot_mat.tms[2], domain, order, cutoff_threshold, setting);
	state.cos_taylor(winman_fram_rot_mat.tms[3], domain, order, cutoff_threshold, setting);
}

void rotate(TaylorModelVec<Real> & states, const TaylorModel<Real> & state1, const TaylorModel<Real> & state2, const TaylorModelVec<Real> & winman_fram_rot_mat, vector<Interval> & domain, const unsigned int order, const Interval & cutoff_threshold) {
	
	TaylorModel<Real> tm1;
	state1.mul_ctrunc(tm1, winman_fram_rot_mat.tms[0], domain, order, cutoff_threshold);
	TaylorModel<Real> tm2;
	state2.mul_ctrunc(tm2, winman_fram_rot_mat.tms[1], domain, order, cutoff_threshold);
	states.tms.push_back(tm1 + tm2);
	// x3 = x2 * frame[2] + x3 * frame[3];
	TaylorModel<Real> tm3;
	state1.mul_ctrunc(tm3, winman_fram_rot_mat.tms[2], domain, order, cutoff_threshold);
	TaylorModel<Real> tm4;
	state2.mul_ctrunc(tm4, winman_fram_rot_mat.tms[3], domain, order, cutoff_threshold);
	states.tms.push_back(tm3 + tm4);
}

void normalize(TaylorModelVec<Real> & normalized_states, const TaylorModelVec<Real> & states, const TaylorModel<Real> & state, vector<Interval> & domain, const unsigned int order, const Interval & cutoff_threshold, const Global_Setting & setting) {
	TaylorModel<Real> norm;
	state.rec_taylor(norm, domain, order, cutoff_threshold, setting);
	for(TaylorModel<Real> tm: states.tms) {
		TaylorModel<Real> tm_output;
		tm.mul_ctrunc(tm_output, norm, domain, order, cutoff_threshold);
		normalized_states.tms.push_back(tm_output);
	}
}	 
	
void preprocess(TaylorModelVec<Real> & tmv_inputs, const TaylorModelVec<Real> & winman_fram_rot_mat, const TaylorModelVec<Real> &states, vector<Interval> & domain, const unsigned int order, const Interval & cutoff_threshold, const Global_Setting & setting) {
	// Need to initialize tmv_inputs from outside
	//x2 = tmv_temp.tms[0];
	//x3 = tmv_temp.tms[1];
	//x7 = tmv_temp.tms[2];
	//x10 = tmv_temp.tms[3];
	//x13 = tmv_temp.tms[4];
	//x14 = tmv_temp.tms[5];

	//x2 = x2; x3 = x3;
	//x1 = x1 / 1000.0
	TaylorModel<Real> tm2sq;
	states.tms[0].mul_ctrunc(tm2sq, states.tms[0], domain, order, cutoff_threshold);
	TaylorModel<Real> tm3sq;
	states.tms[1].mul_ctrunc(tm3sq, states.tms[1], domain, order, cutoff_threshold);
	TaylorModel<Real> tm1;
	(tm2sq + tm3sq).sqrt_taylor(tm1, domain, order, cutoff_threshold, setting);
	tmv_inputs.tms.push_back(tm1 / 1000.0);
	// x2 = x2 * frame[0] + x3 * frame[1];
	// x3 = x2 * frame[2] + x3 * frame[3];
	TaylorModelVec<Real> tm23;
	rotate(tm23, states.tms[0], states.tms[1], winman_fram_rot_mat, domain, order, cutoff_threshold);
	TaylorModelVec<Real> tm231;
	normalize(tm231, tm23, tm1, domain, order, cutoff_threshold, setting);
	tmv_inputs.tms.insert(end(tmv_inputs.tms), begin(tm231.tms), end(tm231.tms));
	

	//x4 = sqrt(x5^2 + x6^2)
	//x5 = x1 + rejoin_radius * cos(x14 + PI + region_angle)
	//x6 = x2 + rejoin_radius * sin(x14 + PI + region_angle)
	TaylorModel<Real> tm14(states.tms[5]);
	tm14 += (PI + rejoin_angle);
	TaylorModel<Real> tm5;
	tm14.cos_taylor(tm5, domain, order, cutoff_threshold, setting);
	tm5 += states.tms[0];
	TaylorModel<Real> tm6;
	tm14.sin_taylor(tm6, domain, order, cutoff_threshold, setting);
	tm6 += states.tms[1];
	TaylorModel<Real> tm5sq;
	tm5.mul_ctrunc(tm5sq, tm5, domain, order, cutoff_threshold);
	TaylorModel<Real> tm6sq;
	tm6.mul_ctrunc(tm6sq, tm6, domain, order, cutoff_threshold);
	TaylorModel<Real> tm4;
	(tm5sq + tm6sq).sqrt_taylor(tm4, domain, order, cutoff_threshold, setting);
	// x4 = x4 / 1000.0;
	tmv_inputs.tms.push_back(tm4/ 1000.0);
	
	// x5 = x5 * frame[0] + x6 * frame[1];
	// x6 = x5 * frame[2] + x6 * frame[3];
	TaylorModelVec<Real> tm56;
	rotate(tm56, tm5, tm6, winman_fram_rot_mat, domain, order, cutoff_threshold);
	TaylorModelVec<Real> tm564;
	normalize(tm564, tm56, tm4, domain, order, cutoff_threshold, setting);
	tmv_inputs.tms.insert(end(tmv_inputs.tms), begin(tm564.tms), end(tm564.tms));
	
	// x7 = x7 / 400.0;
	TaylorModel<Real> tm7(states.tms[2]);
	tmv_inputs.tms.push_back(tm7 / 400.0);
 
	// x8 = x8 * frame[0] + x9 * frame[1];
	// x9 = x8 * frame[2] + x9 * frame[3];
	TaylorModel<Real> tm8_;
	states.tms[4].cos_taylor(tm8_, domain, order, cutoff_threshold, setting);
	TaylorModel<Real> tm8;
	tm7.mul_ctrunc(tm8, tm8_, domain, order, cutoff_threshold);
	TaylorModel<Real> tm9_;
	states.tms[4].sin_taylor(tm9_, domain, order, cutoff_threshold, setting);
	TaylorModel<Real> tm9;
	tm7.mul_ctrunc(tm9, tm9_, domain, order, cutoff_threshold);
	TaylorModelVec<Real> tm89;
	rotate(tm89, tm8, tm9, winman_fram_rot_mat, domain, order, cutoff_threshold);
	TaylorModelVec<Real> tm897;
	normalize(tm897, tm89, tm7, domain, order, cutoff_threshold, setting);
	tmv_inputs.tms.insert(end(tmv_inputs.tms), begin(tm897.tms), end(tm897.tms));

	// x10 = x10 / 400.0;
	TaylorModel<Real> tm10(states.tms[3]);
	tmv_inputs.tms.push_back(tm10 / 400.0);

	// x11 = x11 * frame[0] + x12 * frame[1];
	// x12 = x11 * frame[2] + x12 * frame[3];
	TaylorModel<Real> tm11_;
	states.tms[5].cos_taylor(tm11_, domain, order, cutoff_threshold, setting);
	TaylorModel<Real> tm11;
	tm10.mul_ctrunc(tm11, tm11_, domain, order, cutoff_threshold);
	TaylorModel<Real> tm12_;
	states.tms[5].sin_taylor(tm12_, domain, order, cutoff_threshold, setting);
	TaylorModel<Real> tm12;
	tm10.mul_ctrunc(tm12, tm12_, domain, order, cutoff_threshold);
	TaylorModelVec<Real> tm1112;
	rotate(tm1112, tm11, tm12, winman_fram_rot_mat, domain, order, cutoff_threshold);
	TaylorModelVec<Real> tm111210;
	normalize(tm111210, tm1112, tm10, domain, order, cutoff_threshold, setting);
	tmv_inputs.tms.insert(end(tmv_inputs.tms), begin(tm111210.tms), end(tm111210.tms));
}


int main(int argc, char *argv[])
{
	intervalNumPrecision = 600;
	
	cout << "1" << endl;
	
	unsigned int numVars = 10;
	Variables vars;
	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	//int x1_id = vars.declareVar("x1"); // rel_dist_lead
	int x2_id = vars.declareVar("x2");	// rel_x_lead
	int x3_id = vars.declareVar("x3");	// rel_y_lead
	//int x4_id = vars.declareVar("x4");	// rel_dist_rej
	//int x5_id = vars.declareVar("x5"); // rel_x_rej
	//int x6_id = vars.declareVar("x6");	// rel_y_rej
	int x7_id = vars.declareVar("x7");	// wvel
	//int x8_id = vars.declareVar("x8");	// x_wvel
	//int x9_id = vars.declareVar("x9");	// y_wvel
	int x10_id = vars.declareVar("x10");	// lvel
	//int x11_id = vars.declareVar("x11");	// x_lvel
	//int x12_id = vars.declareVar("x12");	// y_lvel

	int x13_id = vars.declareVar("x13");	// hd_w
	int x14_id = vars.declareVar("x14");	// hd_l
 
	cout << "11" << endl;
	
 
	int u1_id = vars.declareVar("u1");	// u1_lacc
	int u2_id = vars.declareVar("u2");	// u2_lacc	
	int u3_id = vars.declareVar("u3");	// u1_wacc
	int u4_id = vars.declareVar("u4");	// u2_wacc

	int domainDim = numVars + 4;
	 
	cout << "111" << endl;
 
	
	//u1 is wingman heading acc, u3 is wingman vel acc, lead accs are 0
	//Expression<Real> deriv_x1("(x2 * (x11 - x8) + x3 * (x12 - x9)) / x1", vars); //  deriv_x0 = (2 * x1 * deriv(x1) + 2 * x2 * deriv(x2))/2sqrt(x1*x1 + x2 * x2)
	Expression<Real> deriv_x2("x10 * cos(x14) - x7 * cos(x13)", vars); //  deriv_x1 = x10 - x7
	Expression<Real> deriv_x3("x10 * sin(x14) - x7 * sin(x13)", vars); // deriv_x2 = x11 - x8
	//Expression<Real> deriv_x4("(x5 * (x11 - x8) + x6 * (x12 - x9)) / x4", vars); // deriv_x3 = (2 * x4 * deriv(x4) + 2 * x5 * deriv(x5))/sqrt(x4*x4 + x5 * x5)
	//Expression<Real> deriv_x5("x11 - x8", vars); // deriv_x4 = x7
	//Expression<Real> deriv_x6("x12 - x9", vars); // deriv_x5 = x8
	Expression<Real> deriv_x7("u3", vars); // deriv_x6 = (2 * x7 * deriv(x7) + 2 * x8 * deriv(x8))/sqrt(x7*x7 + x8 * x8)
	//Expression<Real> deriv_x8("u3 * cos(x13) - u1 * x7 * sin(x13)", vars); // deriv_x7 = u3 * x7/x6 - u2 * x8
	//Expression<Real> deriv_x9("u3 * sin(x13) + u1 * x7 * cos(x13)", vars); // deriv_x8 = u3 * x8/x6 + u2 * x7
	Expression<Real> deriv_x10("0", vars); // deriv_x9 = (2 * x10 * deriv(x10) + 2 * x11 * deriv(x11))/sqrt(x10*x10 + x11 * x11)
	//Expression<Real> deriv_x11("0", vars); // deriv_x10 = u1 * x11/x10 - u1 * x12
	//Expression<Real> deriv_x12("0", vars); // deriv_x11 = u1 * x12/x10 - u1 * x11
	Expression<Real> deriv_x13("u1", vars); // deriv_x12 = u0
	Expression<Real> deriv_x14("0", vars); // deriv_x13 = u2
	
	

	Expression<Real> deriv_u1("0", vars);
	Expression<Real> deriv_u2("0", vars);
	Expression<Real> deriv_u3("0", vars);
	Expression<Real> deriv_u4("0", vars);
	cout << "1111" << endl;

	// Define the continuous dynamics according to 
	

	vector<Expression<Real>> ode_rhs(numVars);
	//ode_rhs[x1_id] = deriv_x1;
	ode_rhs[x2_id] = deriv_x2;
	ode_rhs[x3_id] = deriv_x3;
	//ode_rhs[x4_id] = deriv_x4;
	//ode_rhs[x5_id] = deriv_x5;
	//ode_rhs[x6_id] = deriv_x6;
	ode_rhs[x7_id] = deriv_x7;
	//ode_rhs[x8_id] = deriv_x8;
	//ode_rhs[x9_id] = deriv_x9;
	ode_rhs[x10_id] = deriv_x10;
	//ode_rhs[x11_id] = deriv_x11;
	//ode_rhs[x12_id] = deriv_x12;
	ode_rhs[x13_id] = deriv_x13;
	ode_rhs[x14_id] = deriv_x14;
	

	//ode_rhs[x15_id] = deriv_x15;
	//ode_rhs[x16_id] = deriv_x16;
	//ode_rhs[x17_id] = deriv_x17;
	//ode_rhs[x18_id] = deriv_x18;
	//ode_rhs[x19_id] = deriv_x19;
	//ode_rhs[x20_id] = deriv_x20;
	//ode_rhs[x21_id] = deriv_x21;
	//ode_rhs[x22_id] = deriv_x22;

	ode_rhs[u1_id] = deriv_u1;
	ode_rhs[u2_id] = deriv_u2;
	ode_rhs[u3_id] = deriv_u3;
	ode_rhs[u4_id] = deriv_u4;
	cout << "11111" << endl;
	Deterministic_Continuous_Dynamics dynamics(ode_rhs);

	// Specify the parameters for reachability computation.
	Computational_Setting setting;

	unsigned int order = stoi(argv[4]);
	cout << "order: " << order << endl;
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
	cout << "111111" << endl;
	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	double w = stod(argv[1]); // 0.5
	cout << "w: " << w << endl;

	int steps = stoi(argv[2]);
	cout << "steps: " << steps << endl;

	Interval 
		//init_x13(0, 2 * PI),
		init_x13(0.5 * PI, PI),
		//init_x14(0, 2 * PI),
		init_x14(0.5 * PI, PI),
		//init_x2(7500 - 100, 7500 + 100), 
		//init_x3(7500 - 100, 7500 + 100);
		init_x2(7500 - 10, 7500 + 10), 
		init_x3(7500 - 10, 7500 + 10);
	//Interval 
	//	init_x1(init_x2.pow(2) + init_x3.pow(2));
	//	init_x1.sqrt_assign();
	//Interval
	//	init_x5(init_x2 + (init_x14 + rejoin_angle + PI).cos() * rejoin_radius), 
	//	init_x6(init_x3 + (init_x14 + rejoin_angle + PI).sin() * rejoin_radius);
	//Interval 
	//	init_x4(init_x5 * init_x5 + init_x6 * init_x6);
	//	init_x4.sqrt_assign(); 
	//Interval init_x7(300 - 100, 300 + 100);
	Interval init_x7(300 - 10, 300 + 10);
	Interval 
	//	init_x8(init_x7 * (init_x13.cos())),
	//	init_x9(init_x7 * (init_x13.sin())),
		//init_x10(280 - 100, 280 + 100);
		init_x10(280 - 10, 280 + 10);
	//Interval
	//	init_x11(init_x10 * (init_x14.cos())),
	//	init_x12(init_x10 * (init_x14.sin()));

 
		
	Interval
		init_u1(0),
		init_u2(0),
		init_u3(0),
		init_u4(0); 
		
	std::vector<Interval> X0;
	//X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	//X0.push_back(init_x4);
	//X0.push_back(init_x5);
	//X0.push_back(init_x6);
	X0.push_back(init_x7);
	//X0.push_back(init_x8);
	//X0.push_back(init_x9);
	X0.push_back(init_x10);
	//X0.push_back(init_x11);
	//X0.push_back(init_x12);
	X0.push_back(init_x13);
	X0.push_back(init_x14);
 
 
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
	string nn_name = "./rejoin_tanh64x64_v3";
	
	NeuralNetwork nn(nn_name);
	cout << "Neural network loade: " << nn_name << endl;
	/* Not in the newest version???
	unsigned int maxOrder = 15;
	Global_Computation_Setting g_setting;
	g_setting.prepareForReachability(maxOrder);
	*/

	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-7, 1e-7);
	//Interval cutoff_threshold(-1e-10, 1e-10);
	unsigned int bernstein_order = stoi(argv[3]);
	cout << "bernstein_order: " << bernstein_order << endl;
	unsigned int partition_num = 4000;

	unsigned int if_symbo = stoi(argv[5]);
	cout << "if_symbo: " << if_symbo << endl;

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
		 
 
		

		TaylorModelVec<Real> tmv_temp;
		initial_set.compose(tmv_temp, order, cutoff_threshold);
		
		
		
		TaylorModelVec<Real> winman_fram_rot_mat;
		build_winman_frame_rot_mat(winman_fram_rot_mat, tmv_temp.tms[4], initial_set.domain, order, cutoff_threshold, setting.g_setting);
		/*
		for (TaylorModel<Real> tm: winman_fram_rot_mat.tms) {
			Interval box;
			tm.intEval(box, initial_set.domain);
			cout << "winman_fram_rot_mat intervals: [" << box.inf() << ", " << box.sup() << "]" << endl; 
		}
		*/
		TaylorModelVec<Real> tmv_input;
		preprocess(tmv_input, winman_fram_rot_mat, tmv_temp, initial_set.domain, order, cutoff_threshold, setting.g_setting);
	
		
		for (TaylorModel<Real> tm: tmv_input.tms) {
			Interval box;
			tm.intEval(box, initial_set.domain);
			cout << "Initial nn input interval: [" << box.inf() << ", " << box.sup() << "]" << endl; 
		}
		

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
	plot_setting.setOutputDims("x2", "x3");

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
