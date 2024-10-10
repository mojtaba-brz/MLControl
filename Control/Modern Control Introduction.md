# Modern Control Introduction
Control Problem (simple description): We have a system (plant) and we want to apply some input to it in order to achieve desirable behavior.

## Types of Controllers (An Overview)

### Passive Control
![Drag reduction in trucks](https://tc.canada.ca/sites/default/files/migrated/aerodynamics_eng_fig11.jpg)

### Active Control
![Active Control Example](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxOkrYzmWsXg8lwwrBkjvdaWeBu0FXcRnrLg&s)

### Open Loop
![Open-Loop Control Diagram](https://electronicscoach.com/wp-content/uploads/2019/11/open-loop-control-system-1.jpg)

### Closed Loop (Feedback Control)
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

We know the solution of an ODE like $\dot{x} = Mx$ is $x(t) = e^{Mt} x(0)$. So, the only thing we need to do for our system is to set K such that all eigenvalues of $(A - BK)$ becomes negative.

References : 
* [Control Bootcamp - By Steve Brunton](https://www.youtube.com/watch?v=Pi7l8mMjYVE&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m)

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

$AT = TD \implies D = T^{-1}AT,$ \\\
$x(t) = Tz(t) \implies \dot{x}=T\dot{z}$ \\\
$\implies \dot{z} = T^{-1}ATz=Dz \implies z(t) = e^{Dt}z(0)$ \\\
$\implies x(t) = Te^{Dt}T^{-1}x(0)$

So it seems if we know the eigenvalues of a linear system, we'll know every thing about it!
**Note :** Eigenvectors represent a special space for the linear system where all of its states are Independent.

**MATLAB :** 
```MATLAB
[T, D] = eig(A);
```

References : 
* [Linear Systems - By Steve Brunton](https://www.youtube.com/watch?v=nyqJJdhReiA&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=2)

## Stability and Eigenvalues
Consider $e^{\lambda_{i}t}$ as the response of a linear system and $\lambda_{i} = a + ib$ then we can say
1. if $a < 0 \implies$ system is stable
2. if $a = 0 \implies$ system is neutral
3. if $a > 0 \implies$ system is unstable

$\implies$ Only when **all** real parts of eigenvalues are negative can we say a linear system is internally stable.
![](./Linear%20Control%20Attachements/{8DFD44AC-5107-4156-ABD3-6D6653ED7100}.png)

In real control problems we don't have access to continuos states of a system. What we have are samples of states at specific times. This fact changes our models to from continous time models to discrete time models.

$$x_{k+1} = \tilde{A}x_{k} \qquad\xrightarrow{\text{solution}} \qquad x_{k} = \tilde{A}^{k}x_{0} \implies {k} = \tilde{T}\tilde{D}^{k}\tilde{T}^{-1}x_{0}$$
$$\tilde{A} = e^{\tilde{A} \Delta t} = \tilde{T}e^{\tilde{D}\Delta t}\tilde{T}^{-1}, \qquad \Delta t = t_{k+1} - t_{k}$$

$\implies$ These systems are stable, if **all** eigenvalues of $\tilde{A}$ are less than one.

![alt text](./Linear%20Control%20Attachements/{2FD14F61-6E1E-4156-A87C-7B4B17F053A8}.png)

References : 
* [Stability and Eigenvalues - By Steve Brunton](https://www.youtube.com/watch?v=h7nJ6ZL4Lf0&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m&index=3)

## Linearizing Around a Fixed Point
Linearizing means to transform a non-linear system $\dot{x} = \vec{f}_{(x)}$ to a linear system $\Delta\dot{x} = A\Delta x$. Here is a step by step algorithm :
1. Find a fixed point $\bar{x}$ where $\vec{f}(\bar{x}) = 0$
2. Linearize about $\bar{x}$ using jacobian of $\vec{f}_{(x)}$ : 

$$A = \left. \frac{Df}{Dx} \right|_{x = \bar{x}} = 
\begin{bmatrix}
\frac{df_{1}}{dx_{1}} & \frac{df_{1}}{dx_{2}} & \frac{df_{1}}{dx_{3}} &  \cdots \\\
\frac{df_{2}}{dx_{1}} & \frac{df_{2}}{dx_{2}} & \frac{df_{2}}{dx_{3}} &  \cdots \\\
\vdots & \vdots & \vdots & \vdots
\end{bmatrix}$$

**An Example :**

![Linearizing Example](./Linear%20Control%20Attachements/{7877F39E-7D5B-4B36-AEF2-6F949D99B8E3}.png)

References : 
* [Linearizing Around a Fixed Point - By Steve Brunton](https://www.youtube.com/playlist?list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m)

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

### Arbitrary Eigenvalues
If we consider $u$ as $u = -Kx$, then K can be computed as follows in **MATLAB** :
```MATLAB
K = place(A, B, eigs)
```

### Discrete Time Systems 
Consider the system as : 

$$x_{k+1} = \tilde{A}x_{k} + \tilde{B}u_{k}$$

A way to test if this discrete system is controllable is to apply a unit impulse to it. If all states change due to this actuation, then the system is fully controllable. Otherwise, only those states that change are controllable.
These sequence shows this test :

$$u_{0} = 1, \quad x_{0} = 0$$
$$u_{1} = 0, \quad x_{1} = \tilde{B} $$
$$u_{2} = 0, \quad x_{2} = \tilde{A}\tilde{B} $$
$$u_{3} = 0, \quad x_{3} = \tilde{A}^{2}\tilde{B} $$
$$\vdots \quad\quad\quad\quad \vdots $$
$$u_{n} = 0, \quad x_{n} = \tilde{A}^{n-1}\tilde{B}$$


If all states move due to the unit impulse, then we can say this system is fully controllable.

### Degree of Controllability and Gramians
For a linear system $\dot{x} = Ax + Bu$ the solution is
$$x_{(t)} = e^{At}x_{(0)}+\int_{0}^{t}e^{A(t-\tau)}Bu_{(\tau)}d\tau$$
#### Controllability Gramian
Controllability Gramian matrix at time $t$ is equal to
$$W_{t}=\int_{0}^{t}{e^{A\tau}BB^Te^{A^T\tau}d\tau}$$
The eigenvector of $W_t$ corresponding to the biggest eigenvalue of $W_t$, shows the most controllable direction of the system and it is vise versa for the eigenvector corresponding to the smallest eigenvalue of $W_t$.
$$W_t\xi_i = \lambda_i\xi_i$$
**Note :** In discrete systems --> $W_t \approx \mathcal{C}\mathcal{C}^T$ 

**MATLAB :**
```MATLAB
[U, D, V] = svd(crtb(A, B), 'econ');
```
U is the matrix with columns of $\xi_i$, and D is diagonal matrix with values of squared of eigenvalues of $W_t$.

![](./Linear%20Control%20Attachements/{DFD58474-857B-4374-85B6-0D9D4C06CD1D}.png)

**Note :** A system is **stabilizable** if and only if, all unstable eigenvectors are in controllable space.
**Note 2 :** And in practice we need those with lightly damped nature to be in controllable space as well.

### Popove-Belovitch-Hautus (PBH) Test

![PBH Test](./Linear%20Control%20Attachements/{9617A3AE-61FF-4607-A861-E3D47BC37E62}.png)


### Cayley-Hamilton Theory
A linear algebra gem! It says almost every matrix A satisfies its own characteristic (eigenvalue) equation. 
$det(A - \lambda I) = 0$ \\
$a_0I + a_1\lambda + \cdots + a_{n-1}\lambda^{n-1}= 0$ \\
$\xrightarrow{we\,can\,replace\,\lambda\,by\,A} a_0I + a_1A + \cdots + a_{n-1}A^{n-1}= 0$

Remember the term, $e^{At}$. It can be right as 
$e^{At} = I + At + \frac{A^2t^2}{2} + \cdots$
Since we can rewrite higher terms as a function of $A^0$ to $A^{n-1}$, We can rewrite above series in as follows \\
$e^{At} = \phi_0(t)I + \phi_1(t)A + \phi_2(t)A^2 + \cdots + \phi_{n-1}(t)A^{n-1}$

### Reachability and Controllability
If a state $x\prime$ is reachable, then: \\\
$x\prime = \int_{0}^{t}e^{A(t-\tau)}Bu_{(\tau)}d\tau$ \\\
for some input $u(t)$.

References : 
* [Controllability - By Steve Brunton](https://www.youtube.com/playlist?list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m)

### Inverted Pendulum on a Cart (A Pole Placement Example)
[Link](./Modern%20Control%20Introduction%20Examples/tune_inverte_pedulum_and_cart_controller.m) of the MATLAB script.

Equations of motion :
$\begin{bmatrix}
\ddot{x} \\
\ddot{\theta} \\
\end{bmatrix} = \begin{bmatrix}
(M + m) & -m L cos(\theta) \\\ 
-mL     & m L^2
\end{bmatrix}^{-1}\begin{bmatrix}
F_x \\\ 
m g L sin(\theta)
\end{bmatrix}$

Here is the MATLAB code used for designing the feedback control "$K$",
```MATLAB
clc; clear; close all

syms g m M L theta Fx real
% dyn_f = [x_2dot; theta_2dot]
dyn_f = [(M + m) -m*L*cos(theta); 
         -m*L    m * L^2]^-1 * ...
         [Fx; m * g * L * sin(theta)];


m = 0.1;
M = 2;
L = 0.6;
g = 9.81;
theta = 0;

A = eval([0 1 0 0
          diff(dyn_f(1), 'x') diff(dyn_f(1), 'x_dot') diff(dyn_f(1), 'theta') diff(dyn_f(1), 'theta_dot')
          0 0 0 1
          diff(dyn_f(2), 'x') diff(dyn_f(2), 'x_dot') diff(dyn_f(2), 'theta') diff(dyn_f(2), 'theta_dot')]);
B = eval([0
          diff(dyn_f(1), 'Fx')
          0
          diff(dyn_f(2), 'Fx')]);

A = double(A);
B = double(B);
K = place(A, B, 5*[-1. -1.1 -1.2 -1.3]);
```
Simulink model is shown below:
![Inverted Pendulum on a Cart in Simulink](./Linear%20Control%20Attachements/{18CCC259-C38C-482A-8754-9261C6B64E05}.png)

### Inverted Pendulum on a Cart (A LQR Example)
LQR is an effort to answer this question, *Where is the **best** place for the poles?*
\
\
$J = \int_{0}^{\infty} (x^TQx+u^TRu) dt, \quad \quad Q\text{ is semi-positive-definite and, }~R\text{ is positive definite}$
\
\
$Q$ represents the error penalty and $R$ represents the control effort or the energy controller produces.

To minimize $J$, there is a $u=-Kx$ control law. Here is the MATLAB command for computing the matrix $K$,
```MATLAB
K = lqr(A, B, Q, R);
```
[Link](./Modern%20Control%20Introduction%20Examples/lqr_inverte_pedulum_and_cart_controller.m) of the MATLAB script.

## Observability
A very similar concept to controllability, but for estimating the states of the system using outputs, is observability. Here is the **observability matrix**:

$\mathcal{O} = \begin{bmatrix}
C \\\
CA \\\
CA^2 \\\
\vdots \\\
CA^{n-1}
\end{bmatrix}$

If the observability matrix has full row rank, then the system is fully observable. In MATLAB, the observability matrix can be computed using `obsv(A, B)` command. An Observer for a linear system can be written as follows:

$\hat{\dot{x}} = A\hat{x}+Bu+K_o(y - \hat{y}) \\
\hat{y} = C\hat{x}$

$K_o$ can be designed by solving a pole placement problem or using the **Kalman filter** concept. The Kalman filter is the LQR of observability! It determines $K_o$​ to minimize the estimation error variance in the presence of known Gaussian measurement noise and disturbances variances.

In MATLAB we can determine the optimal $K_o$ for the **Linear Quadratic Estimator** or **Stationary Kalman filter** problem using `lqe` function.
```
Given the system
        .
        x = Ax + Bu + Gw            {State equation}
        y = Cx + Du + v             {Measurements}
 
    with unbiased process noise w and measurement noise v with 
    covariances
 
        E{ww'} = Q,    E{vv'} = R,    E{wv'} = N ,
 
    [L,P,E] = lqe(A,G,C,Q,R,N)  returns the observer gain matrix L
    such that the stationary Kalman filter
        .
        x_e = Ax_e + Bu + L(y - Cx_e - Du)
 
    produces an optimal state estimate x_e of x using the sensor
    measurements y.
```

### Which Measurement Is The Best (an optimal sensor choosing example)
Given the system,
```
clc; clear; close all

syms g m M L theta Fx real
% dyn_f = [x_2dot; theta_2dot]
dyn_f = [(M + m) -m*L*cos(theta); 
         -m*L    m * L^2]^-1 * ...
         [Fx; m * g * L * sin(theta)];

m = 0.1;
M = 2;
L = 0.6;
g = 9.81;
theta = pi;
Fx = 0;

A = eval([diff(dyn_f(1), 'x_dot') diff(dyn_f(1), 'theta') diff(dyn_f(1), 'theta_dot')
          0 0 1
          diff(dyn_f(2), 'x_dot') diff(dyn_f(2), 'theta') diff(dyn_f(2), 'theta_dot')]);
A = double(A);
```

We need to check which matrix $C$ gives us the best observability power for estimating all states.

```
C = [1 0 0];
[U, D, V] = svd(obsv(A, C));
fprintf("\n")
fprintf("C : [%0.2f %0.2f %0.2f] --> diag(D) : [%0.2f %0.2f %0.2f]", C, diag(D))
fprintf("\n")


C = [0 1 0];
[U, D, V] = svd(obsv(A, C));
fprintf("\n")
fprintf("C : [%0.2f %0.2f %0.2f] --> diag(D) : [%0.2f %0.2f %0.2f]", C, diag(D))
fprintf("\n")

C = [0 0 1];
[U, D, V] = svd(obsv(A, C));
fprintf("\n")
fprintf("C : [%0.2f %0.2f %0.2f] --> diag(D) : [%0.2f %0.2f %0.2f]", C, diag(D))
fprintf("\n")
```

The output is:

```
C : [1.00 0.00 0.00] --> diag(D) : [1.00 0.45 0.45]

C : [0.00 1.00 0.00] --> diag(D) : [15.64 1.00 0.00]

C : [0.00 0.00 1.00] --> diag(D) : [15.64 15.61 0.00]
```

So, as you already have been noticed, the only solution that gives us the most observability power on all states is selecting the velocity $\dot{x}$ sensor.

**Note :** LQE + LQR $\implies$ LQG

 ![LQG](./Linear%20Control%20Attachements/{0AA0CC0D-230F-48FE-9E7E-EF7E6025B315}.png)

 ## Robust Control
 In 1978, Doyel shows there no guaranteed stability margin for LQG regulator.

 ![Doyels article](./Linear%20Control%20Attachements/{3C19C0A2-3F94-41D4-A443-1DF628F14107}.png) 

So, LQG solutions, unlike LQR solutions, provide no global system-independent guaranteed robustness properties. Like their more classical colleagues, modern LQG designers are obliged to test their margins for each specific design.

Here is why we need a design methodology that guaranties required robustness properties when the observer is in the loop. 


In practical control problems, our model of the system has not accurate parameters or not modeled non-linear dynamics, it is under the effect of some external disturbances and our feedbacks are noisy.

![alt text](./Linear%20Control%20Attachements/{C8C45876-1589-45D7-B570-BC8AC4FADB6C}.png)

In this situation, the response of an MIMO system is written as follows:

$y = Pu + P_dd \\
y = PK\epsilon + P_dd \\
y = PK(r - y + n) + P_dd \\
y + PKy = PKr + PKn + P_dd \\
y = (I + PK)^{-1}PKr + (I + PK)^{-1}PKn + (I + PK)^{-1}P_dd$ \
\
$L = PK ~~~ \text{open Loop transfer function} \\
T = (I + L)^{-1}L ~~~~ \text{complementary Transfer function} \\
S = (I + L)^{-1} ~~~~ \text{Sensitivity transfer function}$

**Note :** $T + S = I$

$y = Tr + TPKn + SP_dd$

So tracking and noise attenuation are both depend on $T$, disturbance rejection properties are depend on $S$ and $T + S = I$.

