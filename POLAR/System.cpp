#include "System.h"
#include <assert.h>

using namespace flowstar;
using namespace std;
using json = nlohmann::json;

System::System()
{
    
}
    

System::System(string filename)
{
    cout << "Load the system dynamics..." << endl;
    ifstream input(filename);
    
    if (filename.substr(filename.find_last_of(".") + 1) == "json")
    {
        // Parse json
        json j = json::parse(input);
        num_of_states = j["dynamics"]["state_name_list"].size();
//        cout << this->num_of_states << endl;
        num_of_control = j["dynamics"]["control_name_list"].size();
        state_name_list = j["dynamics"]["state_name_list"];
        control_name_list = j["dynamics"]["control_name_list"];
        assert(j["dynamics"]["state_name_list"].size() == j["dynamics"]["ode_list"].size() && "Miss states or ODEs.");
        for (int i = 0; i < j["dynamics"]["ode_list"].size(); i++)
        {
            ode_list.push_back(j["dynamics"]["ode_list"][i]);
        }
//        for (int i = 0; i < num_of_control; i++)
//        {
//            getline(input, line);
//            ode_list.push_back(Expression_AST<Real>("0"));
//        }
        control_stepsize = j["dynamics"]["control_stepsize"];
        cout << "Succeed." << endl;
        
        cout << "Load the neural network controller..." << endl;
        string nn_file = j["neural_network"];
        nn = NeuralNetwork(nn_file);
        cout << "Succeed." << endl;
    }
    else
    {   // Parse txt
        string line;
        // Parse the dynamics
        if (getline(input, line))
        {
        }
        else
        {
            cout << "failed to read file: System dynamics." << endl;
        }
        try
        {
            num_of_states = stoi(line);
        }
        catch (invalid_argument &e)
        {
            cout << "Problem during string/integer conversion!" << endl;
            cout << line << endl;
        }
        
        getline(input, line);
        num_of_control = stoi(line);
        
        for (int i = 0; i < num_of_states; i++)
        {
            getline(input, line);
            state_name_list.push_back(line);
        }
        
        for (int i = 0; i < num_of_control; i++)
        {
            getline(input, line);
            control_name_list.push_back(line);
        }
        
        for (int i = 0; i < num_of_states; i++)
        {
            getline(input, line);
            ode_list.push_back(line);
        }
//        for (int i = 0; i < num_of_control; i++)
//        {
//            getline(input, line);
//            ode_list.push_back(Expression_AST<Real>("0"));
//        }
        getline(input, line);
        control_stepsize = stod(line);
    }
    
}
