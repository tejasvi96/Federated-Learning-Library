# Federated-Learning-Library
A collection of optimization algorithms for federated learning using TensorFlowFederated

We aim to use Federated Learning for user personalization and explore its potential in the application of next word prediction for mobile keyboards. In a typical federated learning setting we have multiple nodes and a central server with data never leaving the nodes and the gradient computations also being performed by the individual nodes. The central server acts as a orchestrator to aggregate the results from different models with aim to learn a robust model which tries to generalize the behaviour of multiple nodes. The major challenges in a typical federated learning setting are - Optimization Algorithms used to aggregate gradients, Ensuring security of data,  faster communication across the devices. 

To adapt federated learning for personalization, we can modify the setting and assume that the clients also maintain a separate model at their device and based on the global model pass on the relevant information and obtain some information to improve its own local model. 

We aim to explore the different optimization algorithms to efficiently and speedily arrive at a robust model. 
To aggregate the gradients from multiple clients following algorithms from literature are explored.

1.) FedSGD- Federated stochastic gradient descent takes all the  clients  for  federated  aggregation  and  each  client performs one epoch of gradient descent and then broadcasts the model to central server.

2.) [FedAvg](https://arxiv.org/abs/1602.05629)- Federated averaging samples a fraction of users for  each  iteration  and  each  client  can  take  several  steps of gradient descent. The weight update is produced by taking average of gradients of the individual client models.

3.) [FedAttn](https://arxiv.org/pdf/1812.07108.pdf)- It uses a similar setting as FedAvg but aggregates the gradients by a weighted sum of gradients instead of simple averaging. These weights are calculated by making use of the norm of the difference between weights of the central model and a client specific model. These norms are then passed through a softmax function to produce a soft weighting for each of the client. The client with higher norm difference has higher influence in updating the weights of central model. 

4.) [FedMed](https://www.mdpi.com/1424-8220/20/14/4048)- It uses a mediator based incentive scheme to decide whether to use a federated averaging or an adaptive averaging.
The choice is made based on a threshold on the  difference of losses between the consecutive epochs. If the loss is higher then adaptive aggregation of gradeints is done. The adaptive weights are calculated by measuring the Shannon Divergence between the weights of the central model and the user specific model which gives a factor for each client. These vector of factors can be passed through softmax to calculate the actual adaptive weights. 


The  algorithms discussed above aim to learn a centralized model faster based on the different datasets but in a practical scenario we want personalized models per user.
To incorporate the personalization of the model, we take motivations from [here](https://arxiv.org/pdf/2003.13461.pdf) , to arrive at a user specific personalized model.
1.) [AdaptiveFL](https://arxiv.org/pdf/2003.13461.pdf)- The algorithm makes use of 3 models per client- global, personalized, private. focuses specifically on the Makes use of an adaptive alpha  which is used as a mixing parameter to obtain the weighting of a relevant personal model. The balance between the global and local models  is governed by a parameter alpha which is associated with the diversity of the local model and the global model. In general, when the local and global data distributions are well aligned, one would intuitively expect that the optimal choice for the mixing parameter would be small to gain more from the data of other devices. On the
other side, when local and global distributions drift significantly, the mixing parameter needs to be close to one to reduce the contribution from the data of other devices on the optimal local model.
# Results
To test different algorithms we use the standard MNIST dataset and compare the convergence rates of these. The problem setup involves the individual clients mimicing the behaviour of the digits with 10 clients each representing different digits. The learning rate was fixed at 0.1 with SGD optimizer for all these experiments. 

| Algorithm | Test Accuracy (After 50 rounds) | 
| ------------- |:-------------:|
| FedSGD | 81.1 |
| FedAvg | 89.1 | 
| FedMed | 87.1 |
| FedAttn | 76.2 | 

To test the personalization we run FedAVG and AdaptiveFL on the synthetic dataset as discussed[here](https://arxiv.org/abs/1812.06127). The results on IID and Non IID dataset looks like this.

![Images.](https://github.com/tejasvi96/Federated-Learning-Library/blob/main/images/FL_8.png?raw=True)

![Images.](https://github.com/tejasvi96/Federated-Learning-Library/blob/main/images/FL_9.png?raw=True)


We can draw the conclusion that the personalized model in AdaptiveFL is able to determine Non IDness of the data and decides on what contributions to take from the global model. 
