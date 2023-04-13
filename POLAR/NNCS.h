/*---
  Contacting the authors:
  names and emails
---*/


#ifndef NNCS_H_
#define NNCS_H_

#include "NeuralNetwork.h"
#include <chrono>



/*
 * class for Neural Network Controlled Systems
 * The dynamics is defined by an ODE x' = f(x,u) such that the control variable(s) u
 * is updated every delta time (including t = 0) by the neural network controller
 */

template <class DATA_TYPE>
class NNCS
{
protected:
	ODE<DATA_TYPE> ode;							// continuous dynamics
	std::vector<unsigned int> control_var_IDs;	// list of control variables
	double delta;								// control stepsize
	
	NeuralNetwork controller;					// the neural network controller

public:
	NNCS(const Variables & vars, const double control_stepsize, const std::vector<std::string> & str_ode, NeuralNetwork & nn);
	~NNCS() {} ;

	bool updateController(const NeuralNetwork & nn);

	void reach(Result_of_Reachability & result, const Flowpipe & initialSet, const unsigned int n, Computational_Setting & setting, PolarSetting & polar_setting, const std::vector<Constraint> & safeSet);
	void reach(Result_of_Reachability & result, const Flowpipe & initialSet, const unsigned int n, Computational_Setting & setting, PolarSetting & polar_setting, const std::vector<Constraint> & safeSet, Symbolic_Remainder & symbolic_remainder);
};



template <class DATA_TYPE>
NNCS<DATA_TYPE>::NNCS(const Variables & vars, const double control_stepsize, const std::vector<std::string> & str_ode, NeuralNetwork & nn)
{
	ode.stateVars = vars;

	Expression<DATA_TYPE> zero(DATA_TYPE(0));
	ode.expressions.resize(vars.size(), zero);

	int colvar_start = nn.get_num_of_inputs();
	int colvar_end = colvar_start + nn.get_num_of_outputs();

	for(int i=colvar_start; i<colvar_end; ++i)
	{
		control_var_IDs.push_back(i);
	}

	for(int i=0; i<str_ode.size(); ++i)
	{
		Expression<DATA_TYPE> expr(str_ode[i], ode.stateVars);
		ode.expressions[i] = expr;
	}

	controller = nn;

	delta = control_stepsize;
}

template <class DATA_TYPE>
bool NNCS<DATA_TYPE>::updateController(const NeuralNetwork & nn)
{
	// the new controller should use the same interface as the current one
	controller = nn;
}

template <class DATA_TYPE>
void NNCS<DATA_TYPE>::reach(Result_of_Reachability & result, const Flowpipe & initialSet, const unsigned int n, Computational_Setting & setting, PolarSetting & polar_setting, const std::vector<Constraint> & safeSet)
{
	Flowpipe step_initial_set = initialSet;

	for(int k=1; k<=n; ++k)
	{
		// computing the control input range
		TaylorModelVec<Real> tmv_state;
		int num_of_state_vars = step_initial_set.tmvPre.tms.size() - control_var_IDs.size();

		// setting the input of the neural network controller
		for(int i=0; i<controller.get_num_of_inputs(); ++i)
		{
			tmv_state.tms.push_back(step_initial_set.tmvPre.tms[i]);
		}

		// computing the neural network output
		TaylorModelVec<Real> tmv_output;

        if(!polar_setting.symb_rem)
        {
            // not using symbolic remainder
        	controller.get_output_tmv(tmv_output, tmv_state, step_initial_set.domain, polar_setting, setting);
        }
        else{
            // using symbolic remainder
        	controller.get_output_tmv_symbolic(tmv_output, tmv_state, step_initial_set.domain, polar_setting, setting);
        }


		for(int i=0; i<tmv_output.tms.size(); ++i)
		{
			step_initial_set.tmvPre.tms[control_var_IDs[i]] = tmv_output.tms[i];
		}


		ode.reach(result, step_initial_set, delta, setting, safeSet);

		if(result.status > 3)
		{
			printf("Feedback: Cannot complete the reachable set computation.\n");
			return;
		}
		else if(result.status == COMPLETED_UNSAFE)
		{
			printf("Feedback: The system is unsafe.\n");
			return;
		}

		step_initial_set = result.fp_end_of_time;

		printf("Step %d: Done.\n", k);
	}
}

template <class DATA_TYPE>
void NNCS<DATA_TYPE>::reach(Result_of_Reachability & result, const Flowpipe & initialSet, const unsigned int n, Computational_Setting & setting, PolarSetting & polar_setting, const std::vector<Constraint> & safeSet, Symbolic_Remainder & symbolic_remainder)
{
	Flowpipe step_initial_set = initialSet;
	double nn_total_time = 0.0, flowstar_total_time = 0.0;
	for(int k=1; k<=n; ++k)
	{
		// computing the control input range
		auto begin = std::chrono::high_resolution_clock::now();
		TaylorModelVec<Real> tmv_state;
		int num_of_state_vars = step_initial_set.tmvPre.tms.size() - control_var_IDs.size();

		// setting the input of the neural network controller
		for(int i=0; i<controller.get_num_of_inputs(); ++i)
		{
			tmv_state.tms.push_back(step_initial_set.tmvPre.tms[i]);
		}

		// computing the neural network output
		TaylorModelVec<Real> tmv_output;

        if(!polar_setting.symb_rem)
        {
            // not using symbolic remainder
        	controller.get_output_tmv(tmv_output, tmv_state, step_initial_set.domain, polar_setting, setting);
        }
        else{
            // using symbolic remainder
        	controller.get_output_tmv_symbolic(tmv_output, tmv_state, step_initial_set.domain, polar_setting, setting);
        }
		auto nn_timing = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(nn_timing - begin);
		double seconds = elapsed.count() *  1e-9;
		printf("nn processing time is %.2f s, ", seconds);
		nn_total_time += seconds;


		for(int i=0; i<tmv_output.tms.size(); ++i)
		{
			step_initial_set.tmvPre.tms[control_var_IDs[i]] = tmv_output.tms[i];
		}

		// computing the flowpipes in one control step

		ode.reach(result, step_initial_set, delta, setting, safeSet, symbolic_remainder);
		auto flowstar_timing = std::chrono::high_resolution_clock::now();
		auto flowstar_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(flowstar_timing - nn_timing);
		seconds = flowstar_elapsed.count() * 1e-9;
		printf("flowstar time is %.2f s.\n", seconds);
		flowstar_total_time += seconds;

		if(result.status > 3)
		{
			printf("Feedback: Cannot complete the reachable set computation.\n");
			return;
		}
		else if(result.status == COMPLETED_UNSAFE)
		{
			printf("Feedback: The system is unsafe.\n");
			return;
		}

		step_initial_set = result.fp_end_of_time;

		printf("Step %d: Done.\n", k);
	}
	printf("NN time: %.2fs, Flow* time: %.2fs.\n", nn_total_time, flowstar_total_time);
}




#endif /* NNCS_H_ */
