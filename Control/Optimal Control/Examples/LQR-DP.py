import numpy as np
from control import dlqr
from time import monotonic_ns

A = np.array([[0.9, 0.01], [-0.2, 0.98]])
B = np.array([[0],[0.5]])

Q = np.ones(np.shape(A))
R = np.array([[0.1]])

t = monotonic_ns() * 1e-9
for i in range(1000):
    K, _, _ = dlqr(A, B, Q, R)
print("DLAR solution : ", K, f", in {(monotonic_ns() * 1e-9 - t)*0.001 : 2.5f} sec")

# To solve LQR problem using DP and value iteration, you must calculate K_next until convergence
# P_next = (A - BK)'P(A - BK) + Q + K'RK
# K_next = (R + B'P_nextB)^-1B'P_nextA

t = monotonic_ns() * 1e-9
for i in range(1000):
    n_itr = 0
    P = np.zeros(np.shape(A))
    K = np.array([[0, 0]])
    while(1):
        P = (A - B @ K).T @ P @ (A - B @ K) + Q + K.T @ R @ K
        pre_K = K
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        
        n_itr += 1
        
        if np.sum(np.abs(pre_K - K)) < 1e-6:
            break

print("DP solution : ", K, "   in ", n_itr, " iterations", f", in {(monotonic_ns() * 1e-9 - t)*0.001} sec")


t = monotonic_ns() * 1e-9
for i in range(1000):
    n_itr = 0
    P = np.zeros(np.shape(A))
    K = np.array([[0, 0]])
    while(1):
        P_pre = P
        for i in range(20):
            P = (A - B @ K).T @ P @ (A - B @ K) + Q + K.T @ R @ K
            if np.sum(np.abs(P_pre - P)) < 1e-4:
                break
        pre_K = K
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        
        n_itr += 1
        
        if np.sum(np.abs(pre_K - K)) < 1e-6:
            break

print("GPI solution : ", K, "   in ", n_itr, " iterations", f", in {(monotonic_ns() * 1e-9 - t)*0.001} sec")