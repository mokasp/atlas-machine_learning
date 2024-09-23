# Policy Gradients
monte carlo policy gradients are a class of reinforcement learning algorithms that optimize the policy directly by leveraging complete episodes of experience. instead of relying on value function estimates, these methods use the actual returns obtained from episodes to update the policy. the primary goal is to adjust the policy parameters to increase the probability of actions that lead to higher cumulative rewards.

## How to Run
make sure to downgrade pyglet to version ```1.5.27``` before running mainfiles
<br/><br/>```pip install pyglet==1.5.27```<br/><br/><br/>to run a mainfile, run
<br/>```./main-files/0-main.py```
## Dependencies
* ```gym==0.7```: For creating and interacting with the FrozenLake environment.
* ```numpy```: For numerical operations and managing the Q-table.
* ```pyglet==1.5.27```: Provides a wide range of functionalities for rendering graphics, handling user input, and managing multimedia content
* ```matplotlib```: For visualizing training progress and performance metrics through plots.
## Functions
* ```update_grad(rewards, grads, alpha, gamma, policy_weights):``` updates the policy weights based on the rewards and gradients collected during an episode.
* ```train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False): ```trains the policy gradient agent by running multiple episodes in the given environment. Optionally renders the environment to visualize the agent's performance.
* ```policy(matrix, weight):``` Computes action probabilities for a given state.
* ```softmax_grad(softmax):``` Computes the gradient of the softmax function.
* ```policy_gradient(state, weight):``` Computes the action and the gradient of the policy for a given state.
## Resources
- https://stackoverflow.com/questions/58585019/understanding-gradient-policy-deriving
- https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d
- https://www.youtube.com/watch?v=KHZVXao4qXs&t=2900s&ab_channel=GoogleDeepMind
