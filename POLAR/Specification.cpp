#include "Specification.h"

using json = nlohmann::json;
using namespace flowstar;
using namespace std;

vector<string> split(const std::string &text, char delim)
{
    vector<string> tokens;
    size_t start = 0, end = 0;
    while ((end = text.find(delim, start)) != string::npos) {
        if (end != start)
        {
            tokens.push_back(text.substr(start, end - start));
        }
        start = end + 1;
    }
    if (end != start)
    {
        tokens.push_back(text.substr(start));
    }
    return tokens;
}

Specification::Specification()
{
    
}
    

Specification::Specification(string filename)
{
    cout << "Load the specification..." << endl;
    ifstream input(filename);
    
    
    if (filename.substr(filename.find_last_of(".") + 1) == "json")
    {
        // Parse json
        json j = json::parse(input);
        
        char delim1 = ':';
        for (int i = 0; i < j["init"].size(); i++)
        {
            vector<string> temp = split(j["init"][i], delim1);
            Interval init_temp(stod(temp[0]),stod(temp[1]));
            init.push_back(init_temp);
//            cout << init_temp << endl;
        }
        
        time_steps = j["time_steps"];
        
        for (int i = 0; i < j["safe_set"].size(); i++)
        {
//            vector<string> temp = split(j["safe_set"][i], delim1);
//            Interval unsafe_temp(stod(temp[0]),stod(temp[1]));
//            safe_set.push_back(unsafe_temp);
            safe_set.push_back(j["safe_set"][i]);
//            cout << safe_set[i] << endl;
        }
//        cout << safe_set[0] << endl;
//        cout << safe_set.size() << endl;
        cout << "Succeed." << endl;
    }
    else
    {
        string line;

        // Parse
        if (getline(input, line))
        {
        }
        else
        {
            cout << "failed to read file: Specification." << endl;
        }
        
        char delim0 = ',';
        
        vector<string> init_string = split(line, delim0);
        
        char delim1 = ':';
        for (int i = 0; i < init_string.size(); i++)
        {
            vector<string> temp = split(init_string[i], delim1);
            Interval init_temp(stod(temp[0]),stod(temp[1]));
            init.push_back(init_temp);
        }
        
        
        getline(input, line);
        time_steps = stoi(line);
        
        getline(input, line);
        vector<string> unsafe_string = split(line, delim0);
        
        for (int i = 0; i < unsafe_string.size(); i++)
        {
            vector<string> temp = split(unsafe_string[i], delim1);
            Interval unsafe_temp(stod(temp[0]),stod(temp[1]));
            unsafe.push_back(unsafe_temp);
        }
    }
}
