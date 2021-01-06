## Action-value methods
Methods for estimating the value of actions and for using estimates to make action selection decision.

true value of action Q<sub>t</sub>(a) = mean reward when action is selected.

below is explanation and pa rameters for action_value_methods.

#### 1. Sample Average Method
One natural value to estimate true value of action is by averaging the awards actually received.

<img src = https://latex.codecogs.com/png.latex?%5Cequiv%20%5Cfrac%7Bsum-of-rewards-when-a-taken-prior-to-t%7D%7Bnumber-of-times-a-taken-prior-to-t%7D>

<img src="https://latex.codecogs.com/gif.latex?%5Cequiv%20%5Cfrac%7B{\sum_{i=1}^{t-1}R_t\ * {1}_{A_t=a}}%7D{\sum_{i=1}^{t-1} \ * {1}_{A_t=a}">

where 1<sub>predicate</sub> denotes random variable that 1 if predicate is true and 0 if it is not.

the simplest action rule is to select one of the action with highest estimated value. i.e greedy policy.

If there is more than one greedy action then randomly one of greedy action is chosen.

A<sub>t</sub> = argmax<sub>a</sub> Q<sub>t</sub>(a)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; i.e. greedy action

Greedy action will waste no time and pick the action with maximum reward.One can behave greedy all the time. However, it is good
idea not to go greedy at all time steps i.e. not always exploit but also explore which is achieved here by choosing random action every once in a while.

#### Parameters:

parameters for GreedyArmedBandit:

k        : number of arms or option (int)</br>
epsilon  : greedy/biased (float) must be between 0 and 1.</br>
n_iter   : number of iterations (int).</br>
means    : list/np.array of mean of distribution in length should be equal to arms or options </br>


#### Conclusion
If the variance is smaller like zero greedy methods would outperform sample average method.

If the variance is larger, then it is good strategy to explore often.
