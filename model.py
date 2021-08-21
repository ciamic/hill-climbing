import numpy as np

class Policy:

    """Policy Model."""

    def __init__(self, s_size=4, a_size=2):
        self.w = 1e-4*np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space
        
    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x)/sum(np.exp(x))
    
    def act(self, state):
        probs = self.forward(state)
        #action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)              # option 2: deterministic policy
        return action

    def update_policy_weights(self, noise_scale):
        
        """Updates the network weights given a noise parameter.
        Params
        ======
            noise_scale (float): noise scale factor for adjusting weights
        """
        
        self.w += noise_scale * np.random.rand(*self.w.shape)
        
    def step_back_policy_weights(self, weights, noise_scale):
        
        """Restores previous network weights and updates search radius with noise parameter.
        Params
        ======
            noise_scale (float): noise scale factor for adjusting search radius
        """
        
        self.w = weights + noise_scale * np.random.rand(*self.w.shape)
        
    def evaluate_policy_single_episode(self, env, gamma, max_t):
        
        """Evaluates the current policy on a given environment. 
        Params
        ======
            env (gym environment): the test environment
            gamma (float): gamma discount factor
            max_t (int): maximum number of steps per episode
            
        """
        
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action = self.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break      
        
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        return rewards, R