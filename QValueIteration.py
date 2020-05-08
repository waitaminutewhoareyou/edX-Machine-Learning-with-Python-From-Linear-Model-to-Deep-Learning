import numpy as np
state_set = list(range(6))
action_set = ["c", "m"]


def reward_function(state, action, next_state):
    reward = {}
    for cur_state in state_set:
        for action in action_set:
            for next_state in state_set:
                if cur_state != next_state:
                    reward[cur_state, action, next_state] = np.abs((next_state-cur_state))**(1/3)
                if (cur_state != 0) and (next_state == cur_state):
                    reward[cur_state, action, next_state]= (cur_state+4)**(-1/2)
                if (cur_state == 0) and (next_state == 0):
                    reward[cur_state, action, next_state] = 0
    return reward
def build_transition_matrix(state_set, action_set):
    num_state, num_action = len(state_set), len(action_set)
    #T = np.zeros((num_state, num_action, num_state))
    T = {}
    for action in action_set:
        for cur_state in state_set:
            for next_state in state_set:
                T[cur_state, action, next_state] = 0
    for cur_state in state_set:
        if cur_state in [1, 2, 3]:
            T[cur_state, "m", cur_state-1] = 1
            T[cur_state, "c", cur_state+2] = 0.7
            T[cur_state, "c", cur_state] = 0.3
        elif cur_state == 0:
            T[cur_state, "m", cur_state] = 1
            T[cur_state, "c", cur_state] = 1
        else:
            T[cur_state, "m", cur_state-1] = 1
            T[cur_state, "c", cur_state] = 1
    return T
transition_p = build_transition_matrix(state_set, action_set)
# for state in state_set:
#     for action in action_set:
#         res = 0
#         for next_state in state_set:
#             res += transition_p[state,action, next_state]
#         print(state, action, res)


def updateQ(state_set, action_set, T=100):
    gamma = 0.6
    Q = np.zeros((len(state_set), len(action_set)))
    reward = reward_function(state_set, action_set, state_set)
    for iter in range(T):
        Q_new = np.zeros((len(state_set), len(action_set)))
        for state in state_set:
            for action in action_set:
                update = 0
                for next_state in state_set:
                    update += transition_p[state, action, next_state]*(
                            reward[state, action, next_state] + gamma*max(Q[next_state, :]))
                if action == "c":
                    Q_new[state, 0] = update
                elif action == "m":
                    Q_new[state,1] = update

        Q = Q_new
    return Q

Q = updateQ(state_set,  action_set, 1)
print(np.round(Q, 3))

