import numpy as np
import control 
from NoModelRLUtilities import *

A = np.array([[-0.0665, 11.5, 0, 0],
              [0, -2.5, 2.5, 0], 
              [-9.5, 0, -13.736, -13.736], 
              [0.6, 0, 0, 0]])

B = np.array([[0,],
              [0,],
              [13.736,],
              [0,]])

Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

R = np.array([[0.1,]])

K, S, E = control.lqr(A, B, Q, R)
print("LQR solution : ", K)

K = np.zeros(B.T.shape)
n_x = A.shape[0]
n_u = B.shape[1]
n = n_x + n_u
sim_steps = 20 # at least we need n*(n+1)/2 data
for dt in (0.1, 0.01, 0.001, 0.0001):
    for _ in range(50):
        Z_bar     = np.zeros((sim_steps, n*(n+1)//2))
        R_bar     = np.zeros((sim_steps, 1))
        for k in range(sim_steps) :
            x_pre = np.random.rand(n_x , 1)
            u_pre = -K @ x_pre + np.random.rand() # a white noise added to make samples independent
            x = (A @ x_pre + B @ u_pre) * dt + x_pre
            u = -K @ x
            
            z_bar = get_z_bar(x_pre, u_pre) - get_z_bar(x, u)
            Z_bar[k] = z_bar.T
            R_bar[k] = (- x_pre.T @ Q @ x_pre - u_pre.T @ R @ u_pre) * dt
            
        P_bar = np.linalg.inv(Z_bar.T @ Z_bar) @ Z_bar.T @ R_bar

        P = get_H_from_H_bar(P_bar)

        P_ux = P[n_x:, :n_x]
        P_uu = P[n_x:, n_x:]
        
        pre_K = K
        K = np.linalg.inv(P_uu) @ P_ux
        
        if np.sum(np.abs(pre_K - K)) < 1e-2:
            break
        
        print("Q-Learning iteration : ", K)
    
    print("Q-Learning solution : ", K, " for dt = ", dt)