import numpy as np
from collections import defaultdict

def policy_evaluation_step(policy, V, state, model, discount_factor):
    """
    Calculates the value of a state under a given policy (one Bellman backup).
    V(s) = E[R + gamma * V(s')]
    """
    v = 0
    # The policy can be deterministic or stochastic
    for action, action_prob in enumerate(policy[state]):
        if action_prob > 0:
            for prob, next_state, reward, done in model[state][action]:
                v += action_prob * prob * (reward + discount_factor * V[next_state])
    return v

def policy_evaluation(policy, model, discount_factor=0.99, theta=1e-8):
    """
    Iteratively evaluates a policy until convergence.
    """
    V = defaultdict(float)
    while True:
        delta = 0
        for state in model:
            v = V[state]
            V[state] = policy_evaluation_step(policy, V, state, model, discount_factor)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def policy_improvement(policy, V, model, discount_factor):
    """
    Improves a policy by acting greedily w.r.t. a value function.
    """
    policy_stable = True
    for state in V.keys():
        old_action_probs = policy[state].copy()
        
        # Find the best action
        action_values = np.zeros(len(model[state]))
        for action in model[state]:
            for prob, next_state, reward, done in model[state][action]:
                action_values[action] += prob * (reward + discount_factor * V[next_state])
        
        # Find best action, breaking ties randomly
        best_action = np.random.choice(np.flatnonzero(action_values == action_values.max()))
        
        # Create a new deterministic policy
        new_action_probs = np.zeros_like(old_action_probs)
        new_action_probs[best_action] = 1.0
        policy[state] = new_action_probs
        
        if not np.array_equal(old_action_probs, new_action_probs):
            policy_stable = False
            
    return policy, policy_stable

def value_iteration(env, discount_factor=0.99, theta=1e-8):
    """
    The Value Iteration algorithm - finds the optimal value function directly.
    """
    V = defaultdict(float)
    states = list(env.model.keys())  # Fixed iteration order

    while True:
        delta = 0
        for state in states:
            v = V[state]
            # Bellman optimality backup: V(s) = max_a sum [P(s'|s,a) * (R(s,a,s') + gamma * V(s'))]
            action_values = []
            for action in env.model[state]:
                action_value = 0
                for prob, next_state, reward, done in env.model[state][action]:
                    action_value += prob * (reward + discount_factor * V[next_state])
                action_values.append(action_value)
            V[state] = max(action_values) if action_values else 0.0
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    return V

def extract_policy_from_value_function(V, env, discount_factor=0.99):
    """
    Extract the optimal policy from the optimal value function.
    """
    policy = defaultdict(lambda: np.zeros(len(env.action_space)))
    
    for state in env.model:
        action_values = np.zeros(len(env.action_space))
        for action in env.model[state]:
            for prob, next_state, reward, done in env.model[state][action]:
                action_values[action] += prob * (reward + discount_factor * V[next_state])
        
        # Find best action, breaking ties randomly
        best_action = np.random.choice(np.flatnonzero(action_values == action_values.max()))
        policy[state][best_action] = 1.0
    
    return policy

def policy_iteration(env, discount_factor=0.99):
    """
    The full policy iteration algorithm.
    """
    # Start with a random policy
    policy = defaultdict(lambda: np.ones(len(env.action_space)) / len(env.action_space))
    
    while True:
        V = policy_evaluation(policy, env.model, discount_factor)
        policy, policy_stable = policy_improvement(policy, V, env.model, discount_factor)
        
        if policy_stable:
            break
            
    return policy, V
