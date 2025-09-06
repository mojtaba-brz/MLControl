def calc_G_t_array(reward_array, gamma):
    G_t_array = []
    for i in range(len(reward_array)):
        G_t = 0
        for reward in reward_array[i:]:
            G_t = reward + gamma * G_t
        G_t_array.append(G_t)
        
    return G_t_array