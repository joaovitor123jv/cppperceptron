#include <iostream>
#include <vector>
#include <cassert>
#include "NeuralNetwork.hpp"
#include "TrainingData.hpp"

using namespace std;

double Neuron::eta = 0.15;    // taxa de aprendizado da rede

double Neuron::alpha = 0.5;   // o bendito do "momentum"
double NeuralNetwork::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) 
    {
        cout << v[i] << " ";
    }

    cout << endl;
}

void manipulaEntradaComArquivo(const char *argv[])
{
	cout<<"Examinando arquivo |"<<argv[1]<<"|"<<endl;

	TrainingData trainData(argv[1]);

	vector<unsigned> topology;
	trainData.getTopology(topology);

	NeuralNetwork myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	while (!trainData.isEof()) 
	{
		++trainingPass;
		cout << endl << "Passo : " << trainingPass;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) 
		{
			break;
		}
		showVectorVals(": Entradas:", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual output results:
		myNet.getResults(resultVals);
		showVectorVals(":  Saída(s):", resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals(":  Alvo:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how well the training is working, average over recent samples:
		cout << "Média de erro da rede: " << myNet.getRecentAverageError() << endl;
	}
}


int main(const int argc, const char * argv[])
{
    if (argc <= 1)
    {
        cout<<"AVISO: RECOMENDADO arquivo de entrada"<<endl;
		cout<<"LOG: Construindo interface de entrada interativa ainda"<<endl;
		cout<<"LOG: Nada será feito"<<endl;

		cout<<"Digite a topologia da rede (exemplo: 2 3 3 1)"<<endl;
		// string entrada;
		// cin>>entrada;
        //
		// stringstream ss(entrada);
        //
		// vector<unsigned> topologia;
		//
		// while(!ss.eof())
		// {
		// 	unsigned n;
		// 	ss >> n;
		// 	topologia.push_back(n);
		// }
        //
		// NeuralNetwork rede(topologia);// Cria uma rede neural com a topologia indicada

    }
    else if(argc == 2)
    {
		manipulaEntradaComArquivo(argv);
		cout<<"Rede treinada."<<endl;
		bool interacaoAtivada;
		do
		{
			cout<<"Deseja entrar algum dado para testar a saída? (S/N)"<<endl;
			string entrada;
			cin>>entrada;
			if(entrada.compare("s") == 0 || entrada.compare("S") == 0) { interacaoAtivada = true; }
			else if (entrada.compare("n") == 0 || entrada.compare("N") == 0) { interacaoAtivada = false; break; }
			else { interacaoAtivada = false; }
			
			cout<<"Digite a entrada pros neurônios"<<endl;
			cin>>entrada;
			cout<<"Recebido = |"<<entrada<<"|"<<endl;
			stringstream ss(entrada);
			string valor;
			ss>>valor;

		}while(interacaoAtivada);
    }
    else
    {
        cout<<" Argumentos inválidos, verifique os parâmetros"<<endl;
        exit(-1);
    }

    cout << endl << "Done" << endl;
}
