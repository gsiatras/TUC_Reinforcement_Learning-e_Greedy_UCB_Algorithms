# e_Greedy_UCB_Algorithms
Environment:
k stochastic bandits
Each bandit i has a reward that is uniformly distributed in [a_i, b_i].
a_i, b_i should be chosen randomly in [0,1] and must be different for each arm i.
Example, if a_i = 0.3 and b_i = 0.8 for arm i, then its reward at a given time can take any value in [0.3,0.8] with equal probability ("uniform") and its expected reward mu_i = 0.55
Algorithms:
ε-Greedy: assume ε_t gets reduced according to the theorem in the slides.
Upper Confidence Bound algorithm
Measurement Tasks:
Produce plots that prove or disprove the respective sublinear regret rates for each scheme. It's up to you to decide what should suffice here.
Compare the convergence/learning speed of the two algorithms for T = 1000, k = 10
Repeat  (2) for another two scenarios with different T,k values and comment on the differences similarities.
