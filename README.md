# DMF-TONN
Direct Mesh-free Topology Optimization using Neural Networks

This repository contains code for the paper **DMF-TONN: Direct Mesh-free Topology Optimization using Neural Networks** by Aditya Joglekar, Hongrui Chen, Levent Burak Kara

**Abstract:**
We propose a direct mesh-free method for performing topology optimization by integrating a density field approximation neural network with a displacement field approximation neural network. We show that this direct integration approach can give comparable results to conventional topology optimization techniques, with an added advantage of enabling seamless integration with post-processing software, and a potential of topology optimization with objectives where meshing and Finite Element Analysis (FEA) may be expensive or not suitable. Our approach (DMF-TONN) takes in as inputs the boundary conditions and domain coordinates and finds the optimum density field for minimizing the loss function of compliance and volume fraction constraint violation. The mesh-free nature is enabled by a physics-informed displacement field approximation neural network to solve the linear elasticity partial differential equation and replace the FEA conventionally used for calculating the compliance. We show that using a suitable Fourier Features neural network architecture and hyperparameters, the density field approximation neural network can learn the weights to represent the optimal density field for the given domain and boundary conditions, by directly backpropagating the loss gradient through the displacement field approximation neural network, and unlike prior work there is no requirement of a sensitivity filter, optimality criterion method, or a separate training of density network in each topology optimization iteration.

**Summary Figure:**
![Method](https://user-images.githubusercontent.com/92458082/235986996-c52e4265-c005-4e9c-a421-653deaebaa1c.png)
