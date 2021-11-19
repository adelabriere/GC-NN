This project centralize various appplications of graph neural networks as a benchmark applications

# Project 1: Q-SAR prediction, attentive_fp

The main goal of this folder is to determine the usefulness of attentive_fp on the RT prediction task. The state ofthe art is that th ebest-in-class algorithms achieve a mean error of 25s. The model proposed attentive_fp achieve an accuracy of 10s. You can run the training with
```

```

# Project 2: A GAN for graph generation
This project is multi components:
* graph_policy: The graph policy folder aim at finding a decomposition of a molecule into a set of policy. Each policy is a vectr stating how the graph should be expanded, the current vector includes the addition of a node ocnnected to any previous node, or the addition of an edge ocnnectedot anothe edge
* policy_prediction: The goal is simple, using the previously produced graph_policy, learn how to produce the next step of the policy using a graph. This will then be used as a first training step in the policy prediction strategy.
* GAN: The GAN will make use of the policy_prediction subprojet at each step to predict, starting from a noise vector a molecule by predicting its policy decomposition. The molecule will then be built and a classifier based on AttentiveFP or XGboost with structural elemnts will be computed.

Current status of the different elements of the Project:
