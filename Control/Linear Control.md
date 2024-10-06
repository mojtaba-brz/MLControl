# Linear Control
Control Problem (simple description): We have a system (plant) and we want to apply some input to it in order to achieve desirable behavior.

## Types of Controllers (An Overview)

### Passive Control
![Drag reduction in trucks](https://tc.canada.ca/sites/default/files/migrated/aerodynamics_eng_fig11.jpg)

### Active Control
![Active Control Example](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxOkrYzmWsXg8lwwrBkjvdaWeBu0FXcRnrLg&s)

#### Open Loop
![Open-Loop Control Diagram](https://electronicscoach.com/wp-content/uploads/2019/11/open-loop-control-system-1.jpg)

#### Closed Loop (Feedback Control)
![Closed-Loop Control Diagram](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTE7Y8M7OoHdUErhFZh9_pjk5mcSO5dX6XYOA&s)

Why we need feedback?
1. Uncertainty (Internal)
2. Instability
3. Disturbance (External or Exogenous)
4. More Energy Efficient

This example shows how to stabilize a system using feedback.
Consider the system as 
$$\dot{x} = Ax + Bu$$

if we consider $u = -Kx$, the this system can be written as 
$$\dot{x} = Ax - BKx = (A - BK)x$$

We know the solution of an ODE like $\dot{x} = Mx$ is $x(t) = e^{Mt} x(0)$. So, the only thing we need to do for our system is to set K such that all eigenvalues of $(A - BK)x$ becomes negative.

References : 
* [Control Bootcamp - By Steven Brunton](https://www.youtube.com/watch?v=Pi7l8mMjYVE&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m)

## Linear Systems
Consider a system as $\dot{x} = Ax$. We know its solution is $x(t) = e^{At}x(0)$. But how can we calculate th term $e^{At}$?
If $A$ was a diagonal matrix, the solution would be simple :

$$e^{\begin{bmatrix}
\lambda_{1} & 0 & \cdots & \cdots & \cdots \\\
0 & \lambda_{2} & 0 & \cdots & \cdots \\\
0 & 0 & \lambda_{3} & 0 & \cdots \\\
\vdots  & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}} = {\begin{bmatrix}
e^{\lambda_{1}} & 0 & \cdots & \cdots & \cdots \\\
0 & e^{\lambda_{2}} & 0 & \dots & \dots \\\
0 & 0 & e^{\lambda_{3}} & 0 & \dots \\\
\vdots  & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}}$$

Using eigenvalues and eigenvectors of $A$, We can reach to a Diagonal for $x(t) = e^{At}x(0)$ formula. Consider

$$A\vec{v_{i}} = \lambda_{i}\vec{v_{i}}$$

Which $\vec{v_{i}}$ is ith eigenvector of $A$ and $\lambda_{i}$ is ith eigenvalue of $A$.
**Note :** If eigenvalues are dependent or complex numbers, it'll get more complex. We're going to address that later.

If can define $T$ and $D$ as

$$T = {\begin{bmatrix}
\vec{v_{1}} & \vec{v_{2}} & \cdots
\end{bmatrix}}, 
D = {\begin{bmatrix}
\lambda_{1} & 0 & \cdots & \cdots & \cdots \\\
0 & \lambda_{2} & 0 & \dots & \dots \\\
0 & 0 & \lambda_{3} & 0 & \dots \\\
\vdots  & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}}$$

And considering $x(t) = Tz(t)$, We can rewrite the solution as follows:

$$AT = TD \implies D = T^{-1}AT, \\\
x(t) = Tz(t) \implies \dot{x}=T\dot{z} \\\
\implies \dot{z} = T^{-1}ATz=Dz \implies z(t) = e^{Dt}z(0) \\\
\implies x(t) = Te^{Dt}T^{-1}x(0)$$ 

So it seems if we know the eigenvalues of a linear system, we'll know every thing about it!
**Note :** Eigenvectors represent a special space for the linear system where all of its states are Independent.

**MATLAB :** 
```MATLAB
[T, D] = eig(A);
```

References : 
* [Linear Systems - By Steven Brunton](https://www.youtube.com/watch?v=nyqJJdhReiA&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=2)

## Stability and Eigenvalues
Consider $e^{\lambda_{i}t}$ as the response of a linear system and $\lambda_{i} = a + ib$ then we can say
1. if $a < 0 \implies $ system is stable
2. if $a = 0 \implies $ system is neutral
3. if $a > 0 \implies $ system is unstable

$\implies$ Only when **all** real parts of eigenvalues are negative can we say a linear system is internally stable.
![](./Linear%20Control%20Attachements/{8DFD44AC-5107-4156-ABD3-6D6653ED7100}.png)

In real control problems we don't have access to continuos states of a system. What we have are samples of states at specific times. This fact changes our models to from continuos time models to discrete time models.

$$x_{k+1} = \tilde{A}x_{k} \qquad\xrightarrow{\text{solution}} \qquad x_{k} = \tilde{A}^{k}x_{0} \implies {k} = \tilde{T}\tilde{D}^{k}\tilde{T}^{-1}x_{0} 
\\\ \\\
\tilde{A} = e^{\tilde{A} \Delta t} = \tilde{T}e^{\tilde{D}\Delta t}\tilde{T}^{-1}, \qquad \Delta t = t_{k+1} - t_{k}$$

$\implies$ These systems are stable, if **all** eigenvalues of $\tilde{A}$ are less than one.
![alt text](./Linear%20Control%20Attachements/{2FD14F61-6E1E-4156-A87C-7B4B17F053A8}.png)

References : 
* [Stability and Eigenvalues - By Steven Brunton](https://www.youtube.com/watch?v=h7nJ6ZL4Lf0&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=3)

## Linearizing Around a Fixed Point
Linearizing means to transform a non-linear system $\dot{x} = \vec{f}_{(x)}$ to a linear system $\Delta\dot{x} = A\Delta x$. Here is a step by step algorithm :
1. Find a fixed point $\bar{x}$ where $\vec{f}(\bar{x}) = 0$
2. Linearize about $\bar{x}$ using jacobian of $\vec{f}_{(x)}$: 
$$A = \left. \frac{Df}{Dx} \right|_{x = \bar{x}} = 
\begin{bmatrix}
\frac{df_{1}}{dx_{1}} & \frac{df_{1}}{dx_{2}} & \frac{df_{1}}{dx_{3}} &  \cdots \\\
\frac{df_{2}}{dx_{1}} & \frac{df_{2}}{dx_{2}} & \frac{df_{2}}{dx_{3}} &  \cdots \\\
\vdots & \vdots & \vdots & \vdots
\end{bmatrix}$$

**An Example :**
![Linearizing Example](./Linear%20Control%20Attachements/{7877F39E-7D5B-4B36-AEF2-6F949D99B8E3}.png)

References : 
* [Linearizing Around a Fixed Point - By Steven Brunton](https://www.youtube.com/watch?v=h7nJ6ZL4Lf0&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=4)

## Controllability
**Controllability** : If a system is controllable, then we can manipulate it such that it reaches from **any state $x(0)$ to any state $x(t)$** (Reachability).
**Note :** In linear systems if a system is controllable it means you can place its eigenvalues any where you desire. It can be checked by checking the column rank of **Controllability Matrix** $\mathcal{C}$.

$$\mathcal{C} = \begin{bmatrix}
B & AB & A^2B & \cdots & A^{n-1}B
\end{bmatrix}
, \quad
n = dimension~of~A$$

If and only of this matrix has column rank of $n$, then the linear system is fully controllable.

**MATLAB :** 
```MATLAB
is_controllable = rank(ctrb(A, B)) == n;
```

**Note :** This criteria just tells us if a system is controllable or not. What if we wanted to know how controllable a system is? The answer lies in **SVD** of **Controllability Matrix** $\mathcal{C}$.

#### Arbitrary Eigenvalues
If we consider $u$ as $u = -Kx$, then K can be computed as follows in **MATLAB** :
```MATLAB
K = place(A, B, eigs)
```

#### Discrete Time Systems 
Consider the system as : 
$$
x_{k+1} = \tilde{A}x_{k} + \tilde{B}u_{k}
$$
A way to test if this discrete system is controllable is to apply a unit impulse to it. If all states change due to this actuation, then the system is fully controllable. Otherwise, only those states that change are controllable.

These sequence shows this test :
$$
u_{0} = 1, \quad x_{0} = 0 \\\
u_{1} = 0, \quad x_{1} = \tilde{B} \\\
u_{2} = 0, \quad x_{2} = \tilde{A}\tilde{B} \\\
u_{3} = 0, \quad x_{3} = \tilde{A}^{2}\tilde{B} \\\
\vdots \quad\quad\quad\quad \vdots \\\
u_{n} = 0, \quad x_{n} = \tilde{A}^{n-1}\tilde{B} 
$$
If all states move due to the unit impulse, then we can say this system is fully controllable.

#### Degree of Controllability and Gramians


#### Popove-Belovitch-Hautus (PBH) Test
![PBH Test](./Linear%20Control%20Attachements/{9617A3AE-61FF-4607-A861-E3D47BC37E62}.png)
References : 
* [Controllability - By Steven Brunton](https://www.youtube.com/watch?v=h7nJ6ZL4Lf0&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=5)
* [Controllability 2 - By Steven Brunton](https://www.youtube.com/watch?v=h7nJ6ZL4Lf0&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=6)
* [Controllability 3 - By Steven Brunton](https://www.youtube.com/watch?v=h7nJ6ZL4Lf0&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=7)
* [Controllability 4 - By Steven Brunton](https://www.youtube.com/watch?v=h7nJ6ZL4Lf0&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=8)
* [Controllability 5 - By Steven Brunton](https://www.youtube.com/watch?v=h7nJ6ZL4Lf0&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=9)

