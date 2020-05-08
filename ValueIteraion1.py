import numpy as np
state_set = list(range(1, 6))
action_set = ["left", "stay", "right"]


def reward(state, action, next_state):
    admissible_set = [(4, "right", 5), (5,"left", 5), (4, "stay", 5), (5, "stay", 5)]
    return 1*((state,action, next_state) in admissible_set)


def build_transition_matrix(state_set, action_set):
    num_state, num_action = len(state_set), len(action_set)
    #T = np.zeros((num_state, num_action, num_state))
    T = {}
    for action in action_set:
        for cur_state in state_set:
            for next_state in state_set:
                T[cur_state, action, next_state] = 0
                if action == "stay":
                    T[cur_state, action, cur_state] = 1/2
                    if cur_state == 1:
                        T[1, action, 2] = 1/2
                    elif cur_state ==5:
                        T[5, action, 4] = 1/2
                    else:
                        T[cur_state, action, cur_state-1] = 1/4
                        T[cur_state, action, cur_state+1] = 1/4
                elif action == "left":
                    T[cur_state, action, cur_state] = 2/3
                    if cur_state == 1:
                        T[cur_state, action, cur_state] = (1/2)
                        T[cur_state, action, cur_state+1] = (1/2)
                    else:
                        T[cur_state, action, cur_state-1] = 1/3
                elif action == "right":
                    T[cur_state, action, cur_state] = 2/3
                    if cur_state == 5:
                        T[cur_state, action, cur_state] = (1/2)
                        T[cur_state, action, cur_state-1] = (1/2)
                    else:
                        T[cur_state, action, cur_state+1] = 1/3
    return T
transition_p = build_transition_matrix(state_set, action_set)


def updateV(state_set, action_set, T=100):
    V = [0 for _ in range(len(state_set))]
    gamma = 0.5
    for iter in range(T):
        for v_state in state_set:
            candiateset = []
            for action in action_set:
                value = 0
                for next_state in state_set:
                    incre = transition_p[v_state, action, next_state] * (
                                reward(v_state, action, next_state) + gamma * V[next_state - 1])
                    value += incre
                candiateset.append(value)
            V[v_state - 1] = max(candiateset)
    return V
