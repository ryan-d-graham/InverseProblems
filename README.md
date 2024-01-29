# InverseProblems
Bayesian and non-Bayesian Inverse Problems

Information on DeepONetTuringInvert.jl:

This script showcases a use case of DeepONet, the most general neural operator architecture invented in 2019 in this paper https://arxiv.org/abs/1910.03193
using Julia's SciML ecosystem. 

In this example a nonlinear pendulum system is formulated and solved using DifferentialEquations.jl. 

The equations governing the toy example are:
x'(t) = v(t);

x''(t) = v'(t) = -gsin(x(t)) - cv(t) + f(t, p)

In this case, the numbers g and c are fixed and not inferred. Instead, the parameters p of the function f(t, p)
are inferred such that what is being done is function-space Bayesian inference. 

In the first part of the code, all library imports are used. 
Model structure and dimension are then set. The Branch and Trunk networks are fully-connected neural network layers.

The arxiv paper and experience both suggest using a large number of training trajectories is necessary to achieve good
generalization performance of the model. The more trajectories, the lower the generalization error, but the higher the
training error will be. 

The number of sensors and discretization points depend on the processes being used as forcing
functions. In this case, a truncated Fourier Series with Gaussian iid coefficients are used to simulate a Stochastic Process
used to apply external torque to the pendulum's swing about the axle "f(t, p)". Uniformly sampled initial angles and frequencies
insure that DeepONet learns how to predict the underlying torques under arbitrary starting conditions, assuming that these 
two values (x0, v0) are provided as the first two elements of the forcing function. In this way, the network has access to the 
ICs of the dynamical system and can train on any trajectory within a specified distribution. 

Once the Model's training loss is low enough (about 2e-3 or less) and once you can recycle the 10000 trajectories without
significantly affecting the loss without further training, it has effectively learned the nonlinear dynamics of the system.

Once sufficiently trained, the Model's forward pass is much faster than an ODE solver which will allow accelerated Bayesian
inference in a PPL such as Turing or Gen. This example uses Turing and the No-U-Turn sampler set at 65% acceptance ratio. 

When I run the script, I typically begin with a learning rate of 1e-2, train in steps of 100-1000 at a time to get a feel
for how quickly the loss is falling, and if it continues to fall significantly for 2-3 runs, I run it for 10000 iterations 
at a time until the loss quits falling. Once it bottoms out, I decrease the LR to 1e-3 and repeat the above procedure. I do
this until it bottoms out again and decrease it to 1e-4. Once a few thousand iterations at this LR are performed and generalization
performance is checked by recycling training solutions and checking the loss for significant increases and finding none, I
continue to the data visualization parts of the code and manually check a handful of traces to verify visually that model
predictions match ground truth. 

Once these checks are done, I move to the Bayesian inference part of the code and run a minimum of 100 iterations of NUTS(0.65)
and check the posterior traces visually for agreement with ground truth. I then re-populate a distribution of solutions by feeding
the posterior distribution of forcing functions into the trained network and a reference ODE solver for comparison. The ground truth
forcing function, solution curve, blue ODE solution distribution and green network solution distribution should all indicate convergence
of the distributions to the ground truth. 

Now have fun with surrogatized function-space Bayesian infernece!
