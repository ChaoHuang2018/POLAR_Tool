#!/usr/bin/python
import sys
import yaml

def main(argv): 
    input_filename = argv[0]


    output_filename = '_'.join(input_filename.split('.yml')[0].split('verisig_models/')[1].split('/'))


    if(len(argv) > 1):
        output_filename = argv[1]

    with open(input_filename, 'r') as f:
        model = yaml.safe_load(f)
        f.close()
    lines = []
        

    assert len(model['weights']) == len(model['activations'])
    wks = model['weights'].keys()
    for i in range(len(wks)):
        assert len(model['weights'][wks[i]]) == len(model['offsets'][wks[i]]) and (len(model['weights'][wks[i]][0]) == len(model['offsets'][wks[i - 1]]) if i >= 1 else True), \
            "{}x{} != {}".format(\
            len(model['weights'][wks[i]]), \
            len(model['weights'][wks[i]][0]), \
            len(model['offsets'][wks[i]]))

    num_of_inputs = len(model['weights'][wks[0]][0])
    lines.append(str(num_of_inputs))

    num_of_outputs = len(model['weights'][wks[-1]])
    lines.append(str(num_of_outputs))

    num_of_hidden_layers = len(model['activations']) - 1
    lines.append(str(num_of_hidden_layers))

    neuron_numbers = []
    for i in range(len(wks) - 1):
        n_i = len(model['weights'][wks[i]]) 
        neuron_numbers.append(n_i)
        lines.append(str(n_i))
    
    activations = []
    for i in range(len(wks)):
        activ = model['activations'][wks[i]]
        if activ == 'Tanh':
            activ = 'tanh'
        elif activ == 'Linear':
            activ = 'Affine'
        elif activ == 'Sigmoid':
            activ = 'sigmoid'
        elif activ == 'ReLU':
            activ = 'relu'
        activations.append(activ)
        lines.append(str(activ))
        

    
    w = [[] for i in range(len(wks))]
    b = [[] for i in range(len(wks))]
    for i in range(len(wks)):
        wid = len(model['weights'][wks[i]][0])
        ht =  len(model['weights'][wks[i]])
        assert ht == len(model['offsets'][wks[i]])
        w[i] = []
        b[i] = []
        for j in range(len(model['weights'][wks[i]])):
            assert wid == len(model['weights'][wks[i]][j])
            w[i].append([])
            for k in range(len(model['weights'][wks[i]][j])):
                wijk = model['weights'][wks[i]][j][k]
                w[i][j].append(wijk)
                lines.append(str(wijk))
            bij = model['offsets'][wks[i]][j]
            b[i].append(bij)
            lines.append(str(bij))
    
    lines.append("0.0")
    lines.append("1.0")
    
    with open(output_filename, 'w') as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
