#include "System.h"

using namespace flowstar;
using namespace std;
using json = nlohmann::json;

System::System()
{
    
}
    

System::System(string filename)
{
    cout << "Parse the system dynamics." << endl;
    ifstream input(filename);
    
    if (filename.substr(filename.find_last_of(".") + 1) == "json")
    {
        // Parse json
        json j = json::parse(input);
        num_of_states = j["num_of_states"];
        num_of_control = j["num_of_control"];
        state_name_list = j["state_name_list"];
        control_name_list = j["control_name_list"];
        for (int i = 0; i < j["ode_list"].size(); i++)
        {
            ode_list.push_back(j["ode_list"][i]);
        }
//        for (int i = 0; i < num_of_control; i++)
//        {
//            getline(input, line);
//            ode_list.push_back(Expression_AST<Real>("0"));
//        }
        control_stepsize = j["control_stepsize"];
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
