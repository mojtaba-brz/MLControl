# Optimal Control
- [Optimal Control](#optimal-control)
  - [Linear Quadratic Regulator (LQR) Problem](#linear-quadratic-regulator-lqr-problem)
    - [Problem Formulation](#problem-formulation)
      - [Continuous-Time System Dynamics](#continuous-time-system-dynamics)
      - [Continuous-Time Cost Function](#continuous-time-cost-function)
    - [Solution Approach for Continuous-Time Systems](#solution-approach-for-continuous-time-systems)
      - [1. Derive the Optimal Control Law](#1-derive-the-optimal-control-law)
      - [2. Solve the Algebraic Riccati Equation (ARE)](#2-solve-the-algebraic-riccati-equation-are)
      - [3. Implement the Control Law](#3-implement-the-control-law)
    - [Discrete-Time Systems](#discrete-time-systems)
      - [Discrete-Time System Dynamics](#discrete-time-system-dynamics)
      - [Discrete-Time Cost Function](#discrete-time-cost-function)
    - [Solution Approach for Discrete-Time Systems](#solution-approach-for-discrete-time-systems)
      - [1. Derive the Optimal Control Law](#1-derive-the-optimal-control-law-1)
      - [2. Solve the Discrete-Time Algebraic Riccati Equation (DARE)](#2-solve-the-discrete-time-algebraic-riccati-equation-dare)
      - [3. Implement the Control Law](#3-implement-the-control-law-1)
    - [Analytical Solution to DARE](#analytical-solution-to-dare)
      - [1. Define the Hamiltonian Matrix](#1-define-the-hamiltonian-matrix)
      - [2. Eigenvalue Decomposition](#2-eigenvalue-decomposition)
      - [3. Partition the Eigenvectors](#3-partition-the-eigenvectors)
      - [4. Compute the Solution $P\_d$](#4-compute-the-solution-p_d)
      - [5. Verify Positive Definiteness](#5-verify-positive-definiteness)
      - [How to Compute SVD of a Matrix](#how-to-compute-svd-of-a-matrix)
        - [Steps to Compute the SVD](#steps-to-compute-the-svd)
          - [1. Form the Matrix $A^T A$](#1-form-the-matrix-at-a)
          - [2. Compute the Eigenvalues and Eigenvectors of $A^T A$](#2-compute-the-eigenvalues-and-eigenvectors-of-at-a)
          - [3. Form the Matrix $V$](#3-form-the-matrix-v)
          - [4. Compute the Singular Values](#4-compute-the-singular-values)
          - [5. Form the Diagonal Matrix $\\Sigma$](#5-form-the-diagonal-matrix-sigma)
          - [6. Compute the Left Singular Vectors](#6-compute-the-left-singular-vectors)
          - [7. Form the Matrix $U$](#7-form-the-matrix-u)
  - [Reinforcement Learning Application](#reinforcement-learning-application)
    - [RL Fundamentals](#rl-fundamentals)
      - [Markov Decision Process (MDP)](#markov-decision-process-mdp)
      - [Value Function $\\mathcal{v}(s)$ - Bellman Equation](#value-function-mathcalvs---bellman-equation)
      - [Action-Value Function $Q(s, a)$](#action-value-function-qs-a)
      - [Policy Evaluation (Prediction)](#policy-evaluation-prediction)
      - [Policy Iteration](#policy-iteration)
      - [Generalized Policy Iteration](#generalized-policy-iteration)
    - [Solving LQR Problem Using Dynamic Programming](#solving-lqr-problem-using-dynamic-programming)
    - [Solving LQR Problem Using Q-Learning For Discrete Time Systems](#solving-lqr-problem-using-q-learning-for-discrete-time-systems)
      - [Estimating $H$ Using Collected Data](#estimating-h-using-collected-data)
    - [Q-Learning For Non-Linear Discrete Time Systems](#q-learning-for-non-linear-discrete-time-systems)
    - [Solving LQR Problem Using Q-Learning For Continuous Time Systems](#solving-lqr-problem-using-q-learning-for-continuous-time-systems)

The objective of optimal control theory is to determine the control signals that will cause a process to satisfy the physical constraints and at the same time minimize (or maximize) some performance criterion.

## Linear Quadratic Regulator (LQR) Problem

The Linear Quadratic Regulator (LQR) is a well-known optimal control technique used to design controllers for linear systems with quadratic cost functions. The LQR problem aims to find the control input that minimizes a quadratic cost function, subject to the dynamics of a linear system.

### Problem Formulation

#### Continuous-Time System Dynamics

Consider a linear time-invariant (LTI) system described by the following state-space equations:

$$
\dot{x}(t) = A x(t) + B u(t)
$$

where:
- $x(t) \in \mathbb{R}^n$ is the state vector.
- $u(t) \in \mathbb{R}^m$ is the control input vector.
- $A \in \mathbb{R}^{n \times n}$ is the system matrix.
- $B \in \mathbb{R}^{n \times m}$ is the input matrix.

#### Continuous-Time Cost Function

The goal is to minimize the following quadratic cost function:

$$
J = \int_{0}^{\infty} \left( x(t)^T Q x(t) + u(t)^T R u(t) \right) dt
$$

where:
- $Q \in \mathbb{R}^{n \times n}$ is a positive semi-definite matrix that penalizes the state deviation.
- $R \in \mathbb{R}^{m \times m}$ is a positive definite matrix that penalizes the control effort.

### Solution Approach for Continuous-Time Systems

The solution to the LQR problem is typically obtained using the following steps:

#### 1. Derive the Optimal Control Law

The optimal control law is given by:

$$
u(t) = -K x(t)
$$

where $K \in \mathbb{R}^{m \times n}$ is the feedback gain matrix. The matrix $K$ is derived using the solution to the algebraic Riccati equation.

#### 2. Solve the Algebraic Riccati Equation (ARE)

The feedback gain matrix $K$ is obtained by solving the following algebraic Riccati equation:

$$
A^T P + P A - P B R^{-1} B^T P + Q = 0
$$

where $P \in \mathbb{R}^{n \times n}$ is a positive definite matrix. The solution $P$ to this equation is used to compute the feedback gain $K$ as follows:

$$
K = R^{-1} B^T P
$$

#### 3. Implement the Control Law

Once the feedback gain $K$ is computed, the optimal control law $u(t) = -K x(t)$ can be implemented in the system.

### Discrete-Time Systems

#### Discrete-Time System Dynamics

Consider a discrete-time linear time-invariant (LTI) system described by the following state-space equations:

$$
x[k+1] = A_d x[k] + B_d u[k]
$$

where:
- $x[k] \in \mathbb{R}^n$ is the state vector at time step $k$.
- $u[k] \in \mathbb{R}^m$ is the control input vector at time step $k$.
- $A_d \in \mathbb{R}^{n \times n}$ is the discrete-time system matrix.
- $B_d \in \mathbb{R}^{n \times m}$ is the discrete-time input matrix.

#### Discrete-Time Cost Function

The goal is to minimize the following quadratic cost function:

$$
J = \sum_{k=0}^{\infty} \left( x[k]^T Q_d x[k] + u[k]^T R_d u[k] \right)
$$

where:
- $Q_d \in \mathbb{R}^{n \times n}$ is a positive semi-definite matrix that penalizes the state deviation.
- $R_d \in \mathbb{R}^{m \times m}$ is a positive definite matrix that penalizes the control effort.

### Solution Approach for Discrete-Time Systems

The solution to the discrete-time LQR problem is typically obtained using the following steps:

#### 1. Derive the Optimal Control Law

The optimal control law is given by:

$$
u[k] = -K_d x[k]
$$

where $K_d \in \mathbb{R}^{m \times n}$ is the feedback gain matrix. The matrix $K_d$ is derived using the solution to the discrete-time algebraic Riccati equation.

#### 2. Solve the Discrete-Time Algebraic Riccati Equation (DARE)

The feedback gain matrix $K_d$ is obtained by solving the following discrete-time algebraic Riccati equation:

$$
A_d^T P_d A_d - P_d - A_d^T P_d B_d (B_d^T P_d B_d + R_d)^{-1} B_d^T P_d A_d + Q_d = 0
$$

where $P_d \in \mathbb{R}^{n \times n}$ is a positive definite matrix. The solution $P_d$ to this equation is used to compute the feedback gain $K_d$ as follows:

$$
K_d = (B_d^T P_d B_d + R_d)^{-1} B_d^T P_d A_d
$$

#### 3. Implement the Control Law

Once the feedback gain $K_d$ is computed, the optimal control law $u[k] = -K_d x[k]$ can be implemented in the discrete-time system.

### Analytical Solution to DARE

The Discrete-Time Algebraic Riccati Equation (DARE) can be solved analytically using the eigenvalue decomposition of the Hamiltonian matrix. The steps are as follows:

#### 1. Define the Hamiltonian Matrix

The Hamiltonian matrix $H$ is defined as:

$$
H = \begin{pmatrix}
A_d & -B_d (R_d + B_d^T P_d B_d)^{-1} B_d^T P_d A_d \\
-Q_d & A_d^T
\end{pmatrix}
$$

#### 2. Eigenvalue Decomposition

Perform an eigenvalue decomposition of the Hamiltonian matrix $H$:

$$
H = V \Lambda V^{-1}
$$

where:
- $\Lambda$ is a diagonal matrix containing the eigenvalues of $H$.
- $V$ is a matrix whose columns are the eigenvectors of $H$.

#### 3. Partition the Eigenvectors

Partition the matrix $V$ into four blocks:

$$
V = \begin{pmatrix}
V_{11} & V_{12} \\
V_{21} & V_{22}
\end{pmatrix}
$$

where:
- $V_{11} \in \mathbb{R}^{n \times n}$
- $V_{12} \in \mathbb{R}^{n \times n}$
- $V_{21} \in \mathbb{R}^{n \times n}$
- $V_{22} \in \mathbb{R}^{n \times n}$

#### 4. Compute the Solution $P_d$

The solution $P_d$ to the DARE is given by:

$$
P_d = V_{21} V_{11}^{-1}
$$

#### 5. Verify Positive Definiteness

Ensure that $P_d$ is positive definite by checking that all eigenvalues of $P_d$ are positive.

#### How to Compute SVD of a Matrix

The Singular Value Decomposition (SVD) is a fundamental matrix factorization technique that decomposes a matrix into three constituent matrices. SVD is widely used in various fields such as signal processing, statistics, and machine learning. Given a matrix $A \in \mathbb{R}^{m \times n}$, the SVD of $A$ is given by:

$$
A = U \Sigma V^T
$$

where:
- $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix whose columns are the left singular vectors of $A$.
- $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing the singular values of $A$.
- $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix whose columns are the right singular vectors of $A$.

##### Steps to Compute the SVD

The SVD of a matrix $A$ can be computed using the following steps:

###### 1. Form the Matrix $A^T A$

Compute the matrix $A^T A$, where $A^T$ is the transpose of $A$:

$$
A^T A \in \mathbb{R}^{n \times n}
$$

###### 2. Compute the Eigenvalues and Eigenvectors of $A^T A$

Find the eigenvalues and eigenvectors of the matrix $A^T A$. Let $\lambda_1, \lambda_2, \dots, \lambda_n$ be the eigenvalues of $A^T A$, and let $v_1, v_2, \dots, v_n$ be the corresponding eigenvectors.

###### 3. Form the Matrix $V$

The matrix $V$ is formed by arranging the eigenvectors $v_1, v_2, \dots, v_n$ as columns:

$$
V = [v_1, v_2, \dots, v_n]
$$

###### 4. Compute the Singular Values

The singular values of $A$ are the square roots of the eigenvalues of $A^T A$:

$$
\sigma_i = \sqrt{\lambda_i} \quad \text{for} \quad i = 1, 2, \dots, n
$$

###### 5. Form the Diagonal Matrix $\Sigma$

The diagonal matrix $\Sigma$ is formed by placing the singular values $\sigma_1, \sigma_2, \dots, \sigma_n$ on its diagonal. If $m \geq n$, $\Sigma$ is an $m \times n$ matrix with the singular values on the diagonal:

$$
\Sigma = \begin{pmatrix}
\sigma_1 & 0 & \cdots & 0 \\
0 & \sigma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma_n \\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{pmatrix}
$$

If $m < n$, $\Sigma$ is an $m \times n$ matrix with the singular values on the diagonal:

$$
\Sigma = \begin{pmatrix}
\sigma_1 & 0 & \cdots & 0 \\
0 & \sigma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma_m
\end{pmatrix}
$$

###### 6. Compute the Left Singular Vectors

The left singular vectors $u_i$ are computed as:

$$
u_i = \frac{1}{\sigma_i} A v_i \quad \text{for} \quad i = 1, 2, \dots, \min(m, n)
$$

If $m > n$, additional left singular vectors can be computed as the orthonormal basis of the null space of $A^T$.

###### 7. Form the Matrix $U$

The matrix $U$ is formed by arranging the left singular vectors $u_1, u_2, \dots, u_m$ as columns:

$$
U = [u_1, u_2, \dots, u_m]
$$

## Reinforcement Learning Application

### RL Fundamentals

#### Markov Decision Process (MDP)

The Markov decision process (MDP) is a mathematical framework used for modeling decision-making problems where the outcomes are partly random and partly controllable. It's a framework that can address most reinforcement learning (RL) problems.

![An MDP](https://ars.els-cdn.com/content/image/1-s2.0-S0952197622002512-gr2.jpg)

#### Value Function $\mathcal{v}(s)$ - Bellman Equation

$$
\mathcal{v}^\pi(s) = E_\pi\{R_t|s_t=s\} = \sum_a [\pi(a|s) \sum_{s^\prime}p(s^\prime, r|s, a)[r + \gamma \mathcal{v}^\pi(s^\prime)]]
$$

#### Action-Value Function $Q(s, a)$

$$
q(s, a) = \sum_{s^\prime}p(s^\prime, r|s, a)[r + \gamma \mathcal{v}^\pi(s^\prime)]
$$

$$
\quad \quad \quad \quad \quad \quad \quad \quad \quad = \sum_{s^\prime}p(s^\prime, r|s, a)[r + \gamma \sum_{a^\prime} \pi(a^\prime|s^\prime)q(s^\prime, a^\prime)]
$$

**Note 1:** In control problems, we usually are looking for deterministic policies, and our environment is deterministic too. So in control problems we usually have:

$$
\mathcal{v}^\pi(s) = r + \gamma \mathcal{v}^\pi(s^\prime)
$$

$$
q(s, a) = r + \gamma q(s^\prime, a^\prime)
$$

#### Policy Evaluation (Prediction)

![Policy Evaluation](./Attachements/{0FBA8A07-3CDD-491C-AA18-F00D26B591EE}.png)

#### Policy Iteration

![PI](./Attachements/{0CD984C9-D907-441B-87D3-D949716B7C68}.png)

![Policy Iteration](./Attachements/{33F6C545-22FE-4C94-B49B-6743E159BBED}.png)

#### Generalized Policy Iteration

![GPI](./Attachements/{60DB7B09-A41D-4649-834B-4F545385D5AC}.png)

### Solving LQR Problem Using Dynamic Programming

**System Equation:**

$$
x_{k+1} = A_d x_k + B_d u_k
$$

To solve the Linear Quadratic Regulator (LQR) problem using Dynamic Programming (DP) and value iteration, one must iteratively compute the control gain matrix $K$ until convergence. The iterative update equations are given by:

$$
P_{t+1} = (A_d - BK_t)' P_t (A_d - BK_t) + Q + K_t' R K_t
$$

$$
K_{t+1} = (R + B_d' P_{t+1} B_d)^{-1} B_d' P_{t+1} A_d
$$

where:
- $P_{t+1}$ is the updated cost-to-go matrix at iteration $t+1$.
- $K_{t+1}$ is the updated control gain matrix at iteration $t+1$.
- $A_d$ and $B_d$ are the system dynamics matrices.
- $Q$ and $R$ are the state and control cost matrices, respectively.
- $P_t$ is the cost-to-go matrix at iteration $t$.
- $K_t$ is the control gain matrix at iteration $t$.

These equations are iterated until $K$ converges to a stable value. For an example, see [LQR-DP.py](./Examples/LQR-DP.py) file.

### Solving LQR Problem Using Q-Learning For Discrete Time Systems

In general form $q^*(x_k, u_k)$ is:

$$
q^*(x_k, u_k) = r + \gamma \argmax_{u_{k+1}}[q(x_{k+1}, u_{k+1})]
$$

In LQR problem, we can define $q(x_k, u_k)$ as:

$$
q(x_k, u_k) = -x_k^T Q x_k - u_k^T R u_k + x_{k+1}^T P x_{k+1}
$$

$$
= \begin{bmatrix}
x_k \\
u_k
\end{bmatrix}^T \begin{bmatrix}
-Q + B_d^T P A_d & B_d^T P A_d \\
A_d^T P B_d & -R + B_d^T P B_d
\end{bmatrix} \begin{bmatrix}
x_k \\
u_k
\end{bmatrix} = z_k^T H z_k
$$

So you can see the action-value function of a linear system can be written as a linear combination of states and inputs. By derivation of $q$ with respect to input $u$, we can compute the optimal steady-state state feedback gain $K$.

$$
q(x_k, u_k) = z_k^T H z_k = z_k^T \begin{bmatrix}
H_{xx} & H_{xu} \\
H_{ux} & H_{uu}
\end{bmatrix} z_k
$$

$$
\frac{\partial{q}}{\partial{u}} = H_{ux} x_k + H_{uu} u_k = 0
$$

$$
u_k = -H_{uu}^{-1} H_{ux} x_k
$$

#### Estimating $H$ Using Collected Data

We know:

$$
z_k^T H z_k = -x_k^T Q x_k - u_k^T R u_k + z_{k+1}^T H z_{k+1}
$$

In order to solve the above equation for H, it is better to reorganize it into the form below:

$$
\bar{z}_k^T\bar{H} = -x_k^T Q x_k - u_k^T R u_k + \bar{z}_{k+1}^T\bar{H} 
$$

$$
(\bar{z}_k - \bar{z}_{k+1})^T\bar{H} = -x_k^T Q x_k - u_k^T R u_k
$$

$
\bar{H} = \begin{bmatrix}
H_{11}\\
 2H_{12}\\
 2H_{13}\\
 \vdots\\
 H_{22}\\
 2H_{23}\\
 \vdots \\
 H_{(n+m)(n+m)}
\end{bmatrix}
$

$
\bar{z} = \begin{bmatrix}
z_{1}^2 \\
z_1z_2 \\
z_1z_3 \\
\vdots \\
z_2^2 \\
z_2z_3 \\
\vdots 
\\z_{(n+m)(n+m)}^2
\end{bmatrix}
$

By gathering all collected data into matrix $\bar{Z}$ and $\bar{R}$, $H$ can be calculated using least square. 
**Note :** Samples must be independent. One way to do so is to use white noise in inputs.

$$
\bar{Z}H = \bar{R} \implies H = (\bar{Z}^T\bar{Z})^{-1}\bar{Z}^T\bar{R}
$$

$
\bar{R} = \begin{bmatrix}
-x_k^T Q x_k - u_k^T R u_k \\
-x_{k + 1}^T Q x_{k + 1} - u_{k + 1}^T R u_{k + 1} \\
\vdots
\end{bmatrix}
$

$
\bar{Z} = \begin{bmatrix}
(\bar{z}_k - \bar{z}_{k+1})^T \\
(\bar{z}_{k+1} - \bar{z}_{k+2})^T \\
\vdots
\end{bmatrix}
$

**Note 2:** To see an example check [LQR-NoModelRL.py](./Examples/LQR-NoModelRL.py).
**Note 3** This is an iterative method so you have to compute $K$ until it converges.

### Q-Learning For Non-Linear Discrete Time Systems
Consider this affine system:

$$
x_{k+1} = f(x_k) + g(x_k)u_k
$$

In general, form of $Q$ is not known. So we need to approximate it. This method is called Adaptive or Approximate Dynamic Programming (ADP). One way to approximate the $Q$ is as follows:

$$
Q(x, u) = W^T\phi(x, u) = W^T\phi(z)
$$

In the function above, $\phi$ represents a kernel or an activation function. The most important consideration when choosing $\phi$ is that it must be such that $Q$ is convex. In the equation above, the only unknown is $W$, which can be estimated through interaction with the environment.

**Note :** The previous section is a specific form of above equation.

After estimating the $W$ you need to solve the below equation to get the optimal $u$:

$$
\frac{\partial}{\partial{u}}Q = \frac{\partial}{\partial{u}}(W^T\phi(x, u)) = 0 \implies u = \cdots
$$

**Note :** This process should be repeated until the policy ($u$) converges.
**Note :** One way to find the $\phi$ is to first add everything you think might be useful to it, then eliminate those with optimal negligible optimal gains ($W$).

### Solving LQR Problem Using Q-Learning For Continuous Time Systems
This problem is not as easy as the discrete case. So usually it is solved partially model free. Consider a system as follows:

$$
\dot{x} = Ax(t) + Bu(t)
$$

For this system LQR-action-value function can be defined as :

$$
Q(x(t), u(t)) = \int_{t}^{\infty}(x(\tau)^TQx(\tau) + u(\tau)^TRu(\tau))d\tau
$$

Analytical solution to the Continuous LQR problem is:

$$
u = -Kx, \quad \quad \quad \quad K = R^-1B^TP
$$

$P$ is calculated by solving this algebraic Riccati equation:

$$
A^TP + PA - PBR^-1B^TP + Q = 0
$$

