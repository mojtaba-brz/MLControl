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
\lambda_{1} & 0 & \cdots & \cdots & \cdots\\
0 & \lambda_{2} & 0 & \cdots & \cdots \\
0 & 0 & \lambda_{3} & 0 & \cdots \\
\vdots  & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}} = 
{\begin{bmatrix}
e^{\lambda_{1}} & 0 & \cdots & \cdots & \cdots\\
0 & e^{\lambda_{2}} & 0 & \dots & \dots \\
0 & 0 & e^{\lambda_{3}} & 0 & \dots \\
\vdots  & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}}$$
Using eigenvalues and eigenvectors of $A$, We can reach to a Diagonal for $x(t) = e^{At}x(0)$ formula. Consider

$$A\vec{v_{i}} = \lambda_{i}\vec{v_{i}}$$

Which $\vec{v_{i}}$ is ith eigenvector of $A$ and $\lambda_{i}$ is ith eigenvalue of $A$.
**Note :** If eigenvalues aren't Independent, it'll be more complex. We're going to address that later.

If can define $T$ and $D$ as

$T = {\begin{bmatrix}
\vec{v_{1}} & \vec{v_{2}} & \cdots
\end{bmatrix}}, 
D = {\begin{bmatrix}
\lambda_{1} & 0 & \cdots & \cdots & \cdots\\
0 & \lambda_{2} & 0 & \dots & \dots \\
0 & 0 & \lambda_{3} & 0 & \dots \\
\vdots  & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}}$

And considering $x(t) = Tz(t)$, We can rewrite the solution as follows:
$$AT = TD \implies D = T^{-1}AT, \\
x(t) = Tz(t) \implies \dot{x}=T\dot{z} \\
\implies \dot{z} = T^{-1}ATz=Dz \implies z(t) = e^{Dt}z(0) \\
\implies x(t) = Te^{Dt}T^{-1}x(0)$$ 
So it seems if we know the eigenvalues of a linear system, we'll know every thing about it!
**Note :** Eigenvectors represent a special space for the linear system where all of its states are Independent.
**MATLAB :** [T, D] = eig(A);
References : 
* [Linear Systems - By Steven Brunton](https://www.youtube.com/watch?v=nyqJJdhReiA&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=2)

## Stability and Eigenvalues

