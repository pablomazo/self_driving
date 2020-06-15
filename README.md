# GeneticCars

The code is made in python and uses the libraries:

	- pygame for the visual part of the videogame
	- pytorch for the part of neural networks


Here we explain how to use this code

Playing the game:

	Execute Play.py file: python Play.py

	Inside Play.py you can select the number and type of players 



Training supervised algorithm:

	Execute train_Supervised.py file: python train_Supervised.py

	It will show three columnsi:

	number of iteration | the error | the relative error with the preavious iteration

	When the training is finished it will save the resulting network in the path: 

	saved_models/final_supervised.pth 



Training genetic algorithm:

	You can train with GUI if you want to see a wonderful animation 
	
	of all the generations. Execute train_Genetic.py 

	You need to give some parameters by konsole 

		- Population: Number of geneticPlayers in each generation

		- Parents: Number of geneticPlayers that are used to create the next generation

		- Permanent parents: Number of parents that will survive to the next generation

		- Hidden layer: Number of neurons in hidden layer

	Example:  python train_Genetic.py 100 5 2 5

	This will create a training with 100 players 5 parents 2 permanents with 5 neurons in the hidden layer

	It will save the neural networks of the best geneticPlayer in the path: 

	saved_models/genetic_<generation>.pth 
 

	If you want to do large trainings you can do the same training without GUI, it will save time

	Execute train_Genetic_no_GUI.py 

	Example: python train_Genetic_no_GUI.py 100 5 2 5
	
	In the same way the train_Genetic.py file if will save the neural network of the best geneticPlayer in the path: 

	saved_models/final_genetic.pth 



Training DQN:

	Execute train_DQN_no_GUI.py file: python train_DQN_no_GUI.py

	It will show a graph with the total reward in each generation

	When the training is finished it will save the neural network of the best performance in the path: 

	saved_models/final_DQN.pth 
	
	If you want to see the agent execute train_DQN.py file: python train_DQN.py
