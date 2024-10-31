import numpy as np

def get_z_bar(x, u):
    n = len(x) + len(u)
    len_total = n*(n+1)//2
    
    z = np.concatenate((x, u))
    
    z_bar = np.zeros((len_total, 1)) 
    z_bar_idx = 0
    for i in range(n):
        for j in range(i, n):
            z_bar[z_bar_idx] = z[i]*z[j]
            z_bar_idx += 1

    return z_bar

def get_H_from_H_bar(H_bar):
    n_total = len(H_bar)
    delta = 1 - 4 * (-2 * n_total)
    n = int((-1 + np.sqrt(delta)) // 2)
    
    H = np.zeros((n, n)) 
    H_bar_idx = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                H[i, j] = H_bar[H_bar_idx, 0]
            else:
                H[i, j] = H_bar[H_bar_idx, 0] * 0.5
                H[j, i] = H_bar[H_bar_idx, 0] * 0.5
            
            H_bar_idx += 1
    
    return H