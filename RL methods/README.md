# Basic Classification of RL Problem & Solution Space

### Model Free vs Model based
##### Model-free:
- connects observation to action (doesn't build model of environment or reward). 
- flow = input(observation) -> Func(some calculation) -> action 
- easier to train.
> current area of active research.


##### Model-based: 
- tries to predict next observation and/or reward.
- takes best action based on this prediction.
- can choose how far to look in future for this prediction.
> deterministic environments like board games.

### Policy-based vs Value-based:
##### Policy-based:
> directly approximated policy of agent i.e. what action should be taken at each step. Policy is probability
> distribution over action space.
##### Value-based:
> Instead of probability distribution, agent calculated value of every possible action and chooses the best.

### Off-policy vs On-policy:
##### Off-policy :
> ability to learn from historical data. 

e.g. from human demonstration, recorded events or previous version of agents.