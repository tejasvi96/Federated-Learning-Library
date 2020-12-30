# Federated-Learning-Library
A collection of optimization algorithms for federated learning using TensorFlowFederated

We aim to use Federated Learning for user personalization and explore its potential in the application of next word prediction for mobile keyboards. In a typical federated learning setting we have multiple nodes and a central server with data never leaving the nodes and the gradient computations also being performed by the individual clients. The central server acts as a orchestrator to aggregate the results from different models with aim to learn a robust model which tries to generalize the behaviour of multiple clients. The major challenges in a typical federated learning setting are - Optimization Algorithms used to aggregate gradients, Ensuring security of data,  faster communication across the devices. 

To adapt federated learning for personalization, we can modify the setting and assume that the clients also maintain a separate model at their device and based on the global model pass on the relevant information and obtain some information to improve its own local model. 

We aim to explore the different optimization algorithms to efficiently and speedily arrive at a robust model. 
To aggregate the gradients from multiple clients following algorithms from literature are explored.

1.) FedSGD-

2.) [FedAvg](https://arxiv.org/abs/1602.05629)-

3.) [FedAttn](https://arxiv.org/pdf/1812.07108.pdf)-

4.) [FedMed](https://www.mdpi.com/1424-8220/20/14/4048)-

To test our algorithms we use the standard MNIST dataset and compare the convergence rates of these. The problem setup involves the indiviudal clients mimicing the behaviour of the digits with 10 clients each representing different digits.

# Results

| Algorithm | Test Accuracy (After 50 rounds) | 
| ------------- |:-------------:|
| FedSGD | 89.1 |
| FedAvg | 87.1 | 
| FedMed | 81.1 |
| FedAttn | 76.2 | 

To incorporate the personalization of the model, we take motivations from [here](https://arxiv.org/pdf/2003.13461.pdf) , to arrive at a user specific personalized model.
This project is still under development phase. 
