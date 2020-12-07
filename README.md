# Self driving agents

Proyect to explore different training algorithms to make an agent complete 
a circuit.

## Dependencies:
- Pygame
- PyTorch
- NumPy
- Matplotlib

## Training:
Three training methodologies are provided:
1. Supervised learning:
```
python train_Supervised.py
```
During execution the epoch, error and relative error with respect to previous iteration
are given.

When train has finished the resulting model is saved to "saved_models/final_supervised.pth "

2. Genetic algorithm:

There are two versions of this code with and without GUI: train_Genetic.py and train_Genetic_no_GUI.py 
which are executed the same way:
```
python train_Genetic.py population parents permanent hidden_size

python train_Genetic_no_GUI.py population parents permanent hidden_size
```
with
* population: Number of individuals in each generation.
* parents: Number of parents that are used to create the next generation.
* permanent: Number of parents that will survive to the next generation.
* hidden_size: Number of neurons in hidden layer.

Example:
```
python train_Genetic.py 100 5 2 5
```
This will create a training with 100 players 5 parents 2 permanents with 5 neurons in the hidden layer

It will save the neural networks of the best genetic player in the path "saved_models/genetic_\<generation>.pth"

3. DQN:

As in the genetic algorith there are two version of this algorithm with and without GUI:
train_DQN.py and train_DQN_no_GUI.py, which are executed the same way:
```
python train_DQN.py

python train_DQN_no_GUI.py
```
Once the model is trained it is saved in "saved_models/final_DQN.pth".

## Play:
To play the game or check your trained agents execute the program "Play.py":
```
python Play.py
```
There are two human players controlled with either keyboard arrows and AWDS.

Also a supervised, genetic and DQN player will be loaded, having those the names:
"best_supervised.pth", "best_genetic.pth" and "best_DQN.pth", located in "saved_models"
folder.

You can try your own models by changing the model file inside Play.py or renaming 
it as one of the above.
