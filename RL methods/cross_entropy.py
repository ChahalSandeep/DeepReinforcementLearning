"""
RL - cross entropy (Model free, Policy Based and on policy)
Strengths: Simplicity and Good Convergence
for problems that don't require complex, multistep policies to be learned and discovered
that has short episodes with frequent rewards.
can be used as a part of larger system or by its own

policy = probability distribution over action space similar to classification problem
cross entropy throws away bad episode and train on better episodes

Algorithm:
1. Play N number of episodes using current model and environment
2. Calculate total reward for every episode and decide reward boundary. Usually use some percentile for all rewards
    such as 50th or 70th
3. Throw away all episodes with reward below boundary
4. Train on remaining episodes using observation as input and actions as output
5. Repeat 1-4 until we are satisfied
"""
