# Temporal Difference Learning
this repository covers three model-free reinforcement learning algorithms that learn to update the value function to maximize an accumulated reward using OpenAI's Gym environment:
1. Monte Carlo Methods
2. Temporal Difference Learning (TD(λ))
3. SARSA(λ)
## Concepts
* bootstrapping: the processes of updating value estimates using previous value estimates instead of exact values from complete episodes
* eligibility trace: mechanism for keeping track of with states/state-action pairs have been visited and can be used to assign credit or blame to states or actions that have positive or negative outcomes. 
* on-policy: an on-policy algorithm iteratively refines a single behavior policy (that generates actions based on an observed state)
* off-policy: an off-policy algorithm maintains both a behavior policy (that generates actions based on an observed state) and a target policy (that is trained with the outcome of the action taken). it is possible to occasionally update the behavior policy using the target policy because off-policy algorithms can learn optimal target policies without regard for the nature of the behavior policy.
* Monte Carlo: type of RL algorithm that is useful in situations where an agent only recieves a reward at the end of an episode. it learns from full episodes of interacting with an environment, and do not bootstrap but instead estimates the value func by averaging observed returns.
* Temporal Difference:type of RL algorithm uses bootstrapping to update the value function using the difference between a current estimate and the one recieved after performing an action, where the value fuction is the expected cumulative reward of a given state. these can be either on or off policy.
* SARSA: this RL algorithm falls under TD learning that is strictly on policy and updates the value estimates by using the rewards and value estimates of the subsequent state and action, keeping in mind the action given by the current policy.
## Dependencies
* ```gym==0.7```: For creating and interacting with the FrozenLake environment.
* ```numpy```: For numerical operations and managing the Q-table.
# Functions
- ```monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):``` performs monte carlo algorithm on a given environment.
- ```td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):``` performs TD(λ) on a given environment.
- ```sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)```: performs SARSA(λ) on a given environment.
- ```ep_greedy_policy(Q, state, epsilon):``` performs epsilon greedy policy
