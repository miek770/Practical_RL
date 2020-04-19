
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    actions = mdp.get_possible_actions(state)

    if action not in actions:
        print("Error: action ", action, " is not possible from state ", state)
        sys.exit()

    Qi = 0.0
    s_ps = mdp.get_next_states(state, action)
    for s_p in s_ps:
        P = mdp.get_transition_prob(state, action, s_p)
        r = mdp.get_reward(state, action, s_p)
        y = gamma
        Vi = state_values[s_p]
        Qi += P * (r + y * Vi)

    return Qi
