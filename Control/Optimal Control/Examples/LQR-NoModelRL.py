import numpy as np
from control import dlqr
from NoModelRLUtilities import *

A = np.array([[0.9, 1.5], [-0.3, 0.8]])
B = np.array([[0], [1]])
Q = np.eye(2, 2)
R = np.eye(1, 1) * .1

K, _, _ = dlqr(A, B, Q, R)
print("DLAR solution : ", K)

print()
K = np.zeros(B.T.shape)
n_x = A.shape[0]
n_u = B.shape[1]
n = n_x + n_u
sim_steps = 10 # at least we need n*(n+1)/2 data
for _ in range(50):
    Z_bar     = np.zeros((sim_steps, n*(n+1)//2))
    R_bar     = np.zeros((sim_steps, 1))
    for k in range(sim_steps) :
        x_pre = np.random.rand(n_x , 1)
        u_pre = -K @ x_pre + np.random.rand() * 0.1 # a white noise added to make samples independent
        x = A @ x_pre + B @ u_pre
        u = -K @ x
        
        z_bar = get_z_bar(x_pre, u_pre) - get_z_bar(x, u)
        Z_bar[k] = z_bar.T
        R_bar[k] = - x_pre.T @ Q @ x_pre - u_pre.T @ R @ u_pre
        
    H_bar = np.linalg.inv(Z_bar.T @ Z_bar) @ Z_bar.T @ R_bar

    H = get_H_from_H_bar(H_bar)

    H_ux = H[n_x:, :n_x]
    H_uu = H[n_x:, n_x:]
    
    pre_K = K
    K = np.linalg.inv(H_uu) @ H_ux
    
    if np.sum(np.abs(pre_K - K)) < 1e-2:
        break
    
    print("Q-Learning iteration : ", K)
    
print("Q-Learning solution : ", K)

# Note : the interesting thing is that it computed an stable gain with white noise actuation