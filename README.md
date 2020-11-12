# Distributed Training with Graph Neural Networks
The objective of this project is to implement distributed training on Graph Neural Networks (GNN). Introduced in 2009, GNNs have become a popular model for representing real time dynamic models especially in areas like computer vision, molecular chemistry and biology, pattern recognition, etc where the data when represented as a graph proved to produce startling results.

However given the architecture of GNNs, training a GNN takes a very long time when done sequentially. Through this project, the authors are attempting to mitigate this issue by introducing distributed computing wherein functionally independent segments of the network can be processed parallely to save on computation time given the system has proper distributed computing support.

### Graph Neural Network models 
For more information checkout https://github.com/CVxTz/graph_classification

#### Baseline ANN model
We first experiment with the simplest model that learn to predict node classes using only the binary features and discarding all graph information.
This model is a fully-connected Neural Network that takes as input the binary features and outputs the class probabilities for each node.

#### Graph Embedding Model


### How to run the code
* Download the [CORA Dataset](https://linqs.soe.ucsc.edu/data) and save it in a folder named **input**
* **python eda.py** to visualise dataset as a graph network
* **python word_features_only.py** for basic ANN model (accuracy ~ 52.4%)
* **python graph_embedding.py** for binary features used as graph network information model (accuracy ~72.2%)
* **python graph_feature_embedding.py** for pretrained binary features used as graph network information model (accuracy ~71.0%)
* **python distributed_graph_feature_embedding.py** for training the model with a distributed mirrored strategy (accuracy ~71.8%)
