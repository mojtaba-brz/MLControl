# Modern Robotics - Mechanics, Planning and Control

- [[#Foundations of Robot Motion|Foundations of Robot Motion]]
	- [[#Foundations of Robot Motion#Configuration Space|Configuration Space]]
		- [[#Configuration Space#Workspace and Taskspace|Workspace and Taskspace]]
		- [[#Configuration Space#Degrees of Freedom of a Rigid Body|Degrees of Freedom of a Rigid Body]]
		- [[#Configuration Space#Grubler’s Formula|Grubler’s Formula]]
		- [[#Configuration Space#Conﬁguration Space: Topology and Representation|Conﬁguration Space: Topology and Representation]]
		- [[#Configuration Space#Some Notes|Some Notes]]
	- [[#Foundations of Robot Motion#Rigid-Body Motions|Rigid-Body Motions]]
		- [[#Rigid-Body Motions#Rotation Matrices|Rotation Matrices]]
		- [[#Rigid-Body Motions#Angular Velocities|Angular Velocities]]
		- [[#Rigid-Body Motions#Exponential Coordinate Representation of Rotation|Exponential Coordinate Representation of Rotation]]
		- [[#Rigid-Body Motions#Homogeneous Transformation Matrices|Homogeneous Transformation Matrices]]
		- [[#Rigid-Body Motions#Twists|Twists]]
		- [[#Rigid-Body Motions#The Screw Interpretation of a Twist|The Screw Interpretation of a Twist]]
		- [[#Rigid-Body Motions#Exponential Coordinate Representation of Rigid-Body Motions|Exponential Coordinate Representation of Rigid-Body Motions]]
		- [[#Rigid-Body Motions#Wrenches|Wrenches]]
	- [[#Foundations of Robot Motion#[Software](https://hades.mech.northwestern.edu/index.php/Modern_Robotics#Software)|[Software](https://hades.mech.northwestern.edu/index.php/Modern_Robotics#Software)]]
- [[#Robot Kinematics|Robot Kinematics]]
	- [[#Robot Kinematics#Forward Kinematics|Forward Kinematics]]
		- [[#Forward Kinematics#Product of Exponentials Formula|Product of Exponentials Formula]]
			- [[#Product of Exponentials Formula#First Formulation: Screw Axes in the Base Frame|First Formulation: Screw Axes in the Base Frame]]
			- [[#Product of Exponentials Formula#Second Formulation: Screw Axes in the End-Eﬀector Frame|Second Formulation: Screw Axes in the End-Eﬀector Frame]]
		- [[#Forward Kinematics#Software|Software]]
- [[#Robot Dynamics|Robot Dynamics]]
- [[#Robot Motion Planning and Control|Robot Motion Planning and Control]]
- [[#Robot Manipulation and Wheeled Mobile Robots|Robot Manipulation and Wheeled Mobile Robots]]
- [[#Capstone Project|Capstone Project]]
- [[#References|References]]


## Foundations of Robot Motion

### Configuration Space
The configuration of a robot is a complete specification of  the position of every point of the robot. 

**Note:** Configuration space (C-space) not only includes the positions of the robot but also its orientations and any constraints that may affect its movement. It represents the complete set of configurations that the robot can achieve, taking into account its joints and any obstacles in the environment.

If a robot's configuration space (C-space) is not properly defined, it can lead to several issues, including:

1. **Inaccurate Simulations**: Without a correct C-space, simulations may not accurately reflect the robot's capabilities or limitations, leading to unexpected behaviors in real-world applications.
    
2. **Collision Detection Problems**: A poorly defined C-space may fail to account for obstacles, resulting in collisions that could damage the robot or its environment.
    
3. **Inefficient Path Planning**: If the C-space does not accurately represent the robot's possible configurations, path planning algorithms may generate suboptimal or impossible paths.

The minimum number n of real-valued coordinates needed to represent the configuration is the number of degrees of freedom (dof) of the robot. The n-dimensional space containing all possible  
configurations of the robot is called the configuration space (C-space). The configuration of a robot is represented by a point in its C-space.

#### Workspace and Taskspace
The **workspace** refers to the physical space that the robot can reach or operate within, based on its configuration and the limitations of its joints. It is indeed influenced by nonholonomic constraints, but it also includes the robot's physical structure and the environment.

On the other hand, the **task space** is a more abstract concept that represents the space in which the robot's end effector (like a gripper or tool) operates. It focuses on the goals of the robot's tasks, such as positions and orientations needed to perform specific actions.

#### Degrees of Freedom of a Rigid Body

$$
Dof = \text{Sum of freedoms of the pints on the object} ~ - ~ \text{Sum of Independent constraints}
$$
Since our robots consist of rigid bodies, Equation above can be expressed as follows:

$$
Dof = \text{Sum of freedoms of the bodies} ~ - ~ \text{Sum of Independent constraints}
$$

#### Grubler’s Formula
If the constraints provided by the joints are independent,

$$
Dof = m(N - 1 - J) + \sum_{i}f_i
$$

$m:\text{number of degrees of freedom of a body in space, 6 for 3D and 3 for 2D}$
$N:\text{Number of bodies including the earth if is attached to the bodies}$
$J:\text{Number of joints}$
$f_i:\text{Number of freedoms of the ith joint}$

**Examples:**

![](Attachements/Pasted%20image%2020250407103921.png)

$Dof = 3(4 - 5) + 5 = 2$

![](Attachements/Pasted%20image%2020250407104821.png)

$Dof = 3(5 - 7) + 7 = 1$

![](Attachements/Pasted%20image%2020250407105557.png)

$Dof = 3((5 + 2) - (6 + 3)) + 6 + 3 = 6 - 9 = 3$

![](Attachements/Pasted%20image%2020250407112437.png)

$Dof = 6(16 - 21) + 9 + 12(3) = 30 - 45 = 15$

#### Conﬁguration Space: Topology and Representation

Representation of basic components of C-space of a robot is described in the table below.

![](Attachements/Pasted%20image%2020250422114910.png)

**For a 2D body:** $\mathbb{R}^2 \times \mathcal{S}^1$
**For a 3D body:** $\mathbb{R}^3 \times \mathcal{S}^2 \times \mathcal{S}^1$

#### Some Notes

- A robot’s C-space can be parametrized explicitly or represented implicitly. For a robot with $n$ degrees of freedom, an explicit parametrization uses $n$ coordinates, the minimum necessary. An implicit representation involves m coordinates with $m \ge n$, with the m coordinates subject to $m - n$ constraint equations. With an implicit parametrization, a robot’s C-space can be viewed as a surface of dimension n embedded in a space of higher dimension m.

- A robot’s task space is a space in which the robot’s task can be naturally expressed. A robot’s workspace is a specification of the configurations that the end-effector of the robot can reach.
- To deform one n-dimensional space into another topologically equivalent space, only stretching is allowed. So an n-dimensional space can **not** be topologically equivalent to an m-dimensional space, where $m≠n$.
- Holonomic constraints are those that can be expressed as equations relating the coordinates of a system, meaning they can be integrated to reduce the number of degrees of freedom. Nonholonomic constraints, on the other hand, cannot be integrated in this way and often involve inequalities or depend on the velocities of the system.

### Rigid-Body Motions
In this chapter we develop a systematic way to describe a rigid body’s position and orientation which relies on attaching a reference frame to the body. The conﬁguration of this frame with respect to a ﬁxed reference frame is then represented as a 4 × 4 matrix. Such a matrix not only represents the conﬁguration of a frame, but can also be used to (1) translate and rotate a vector or a frame, and (2) change the representation of a vector or a frame from coordinates in one frame to coordinates in another frame.

**Note:** All frames in this book are stationary, inertial, frames. When we refer to a body frame { b } , we mean a motionless frame that is instantaneously coincident with a frame that is ﬁxed to a (possibly moving) body.

**Note:** In an n-dimensional space, number of angular degrees of freedom is equal to $\frac{n(n-1)}{2}$. I’m still struggling imagining it!

#### Rotation Matrices
$R$ is a $3\times3$ Matrix subset of **special orthogonal group** $SO(3)$, That is used for:
- To represent an orientation: $R_{ab}$
- To change the reference frame: $R_{ac} = R_{ab}R_{bc}$,   $R_{ab}P_b=p_a$
- To rotate a vector or frame: $Rot(\hat{\omega}, \theta)$

Assume there are two frames, $s$ and $b$. The rotation matrix which between these two frames is defined as:

$$
R_{sb} = \begin{bmatrix} 
\hat{x}^b_s & \hat{y}^b_s & \hat{z}^b_s
\end{bmatrix} = \begin{bmatrix} 
{\hat{x}^s_b}^T \\ {\hat{y}^s_b}^T \\ {\hat{z}^s_b}^T
\end{bmatrix}, ~~~~ R_{bs} = R^T_{sb}
$$

**Note:** Naming Rule for above formula: $Name^{SubName}_{Frame}$. e.g. $R_{sb}$ represents Rotatio of frame { b } relative to frame { s }. For the rest: $Name_{SubName~~or~~Frame}$.

Examples of rotation operations about coordinate frame axes are:

$$
Rot(\hat{x}, \theta) = \begin{bmatrix}
1 & 0 & 0 \\
0 & cos(\theta) & -sin(\theta) \\
0 & sin(\theta) & cos(\theta)
\end{bmatrix}
$$

$$
Rot(\hat{y}, \theta) = \begin{bmatrix}
cos(\theta) & 0 & sin(\theta) \\
0 & 1 & 0 \\
-sin(\theta) & 0 & cos(\theta)
\end{bmatrix}
$$

$$
Rot(\hat{z}, \theta) = \begin{bmatrix}
cos(\theta) & -sin(\theta) & 0 \\
sin(\theta) & cos(\theta) & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

$$
Rot(\hat{\omega}, \theta) = e^{[\hat{\omega}]\theta}
$$

#### Angular Velocities
If we examine the body frame at times t and t+∆t, the change in frame orientation can be described as a rotation of angle ∆θ about some unit axis $\hat{\omega}$ passing through the origin. 

In the limit as ∆t approaches zero, the ratio ∆θ/∆t becomes the rate of rotation $\dot{\theta}$ and $w$ can similarly be regarded as the instantaneous axis of rotation. In fact, $\hat{\omega}$ and $\dot{\theta}$ can be combined to deﬁne the angular velocity $\hat{\omega} \in \mathbb{R}^3$ as follows:

$$
\omega = \hat{\omega}\dot{\theta}
$$

The relation between angular velocity and rotation matrix:
Assume unit vectors of a frame are rotating with angular velocity of $\omega$

$$
\dot{\hat{e_i}} = [\omega]\hat{e_i}, ~~~~~~~~~ [\omega] \in so(3)
$$

and $R$ is defined as

$$
R = \begin{bmatrix}
\hat{e_x} & \hat{e_y} & \hat{e_z}
\end{bmatrix} \xrightarrow{~~~~~~~~~~} \dot{R} = \begin{bmatrix}
\dot{\hat{e_x}} & \dot{\hat{e_y}} & \dot{\hat{e_z}}
\end{bmatrix} = [\omega]R
$$

Which can be summarised as :

$$
[\omega_s] = \dot{R_{sb}}R_{sb}^T, ~~~~~~~ [\omega_b] = R_{sb}^T\dot{R_{sb}}
$$

#### Exponential Coordinate Representation of Rotation
We now introduce a three-parameter representation for rotations, the exponential coordinates for rotation. The exponential coordinates parametrize a rotation matrix in terms of a rotation axis (represented by a unit vector $\hat{\omega}$) and an angle of rotation θ about that axis.

The diﬀerential equation of unit vectors of rotation matrix can be expressed as
$$
\dot{\hat{e_i}}(t) = [\omega]\hat{e_i}(t)
$$
with initial condition $\hat{e_i}(0)$. This is a linear diﬀerential equation of the form $\dot{x} = Ax$ and  its solution is

$$
\hat{e_i}(t) = e^{[\omega]t}\hat{e_i}(0) ~~~~~ \xrightarrow{~~~~ t = \theta/|\omega| ~~~~} ~~~~~ \hat{e_i}(\theta) = e^{[\hat{\omega}]\theta}\hat{e_i}(0)
$$

By expanding the matrix exponential $e^{[\hat{\omega}]\theta}$ in series form

$$
e^{[\hat{\omega}]\theta} = \sum_{i = 0}^{\infty} \frac{([\hat{\omega}]\theta)^i}{i!} = I + \left( \theta - \frac{\theta^3}{3!} + \dots \right)[\hat{\omega}] + \left(\frac{\theta^2}{2} - \frac{\theta^4}{4!} + \dots \right)[\hat{\omega}]^2
$$

$$
e^{[\hat{\omega}]\theta} = I + sin(\theta)[\hat{\omega}] + (1-cos(\theta))[\hat{\omega}]^2
$$

Equation above is also known as Rodrigues’ formula for rotations.

**Note:** $R=e^[\omega]\theta ~~~~ \implies ~~~~~ log R =[\omega]\theta$ ,    if   $tr(R) \ne -1$

**Note:** The orientation of a frame {d} relative to a frame {c} can be represented by a unit rotation axis $\hat\omega$, and the distance $θ$ rotated about the axis. If we rotate the frame {c} by $\theta$ about the axis $\hat\omega$ expressed in the {c} frame, we end up at {d}. The vector $\hat\omega$ has 3 numbers and $\theta$ is 1 number, but we only need 3 numbers, the exponential coordinates $\hat\omega\theta$, to represent {d} relative to {c}, because though we use 3 numbers to represent $\hat\omega$, with, hat, on top, $\hat\omega$, with, hat, on top actually only represents a point in a 2-dimensional space, the 2-dimensional sphere of unit 3-vectors.


#### Homogeneous Transformation Matrices
Rather than identifying Rotation $R$ and translation $p$ separately, we package them into a single matrix as follows.

$$
T = \begin{bmatrix}
R & p \\
0 & 1
\end{bmatrix}_{4\times4} ~~~~\in SE(3) ~~~~~~\text{Special Euclidian group}
$$

- **Inverse of a transformation:**

$$
T^{-1} = \begin{bmatrix}
R^T & -R^Tp \\
0 & 1
\end{bmatrix}
$$

- **homogeneous coordinates:** $[e^T 1]^T$ which is used when we are using T

#### Twists
$\nu$ is a column vector inclding angular and linear velocity (or spatial velocity in the space frame), called twists:.

$$
\nu = \begin{bmatrix}
\omega \\
v
\end{bmatrix} ~~ \in \mathbb{R}^6
$$

matrix representation of the body twist:

$$
T^{-1}_{sb}\dot{T}_{sb}=[\nu_b]=\begin{bmatrix}
[\omega_b] & v_b \\
0 & 0
\end{bmatrix}
$$

$\nu_b$ and $\nu_s$ relation:

$$
\nu_s = \begin{bmatrix}
\omega_s \\
v_s
\end{bmatrix} = \begin{bmatrix}
R_{sb} & 0 \\
[p_{sb}]R_{sb} & R_{sb}
\end{bmatrix}
\begin{bmatrix}
\omega_b \\
v_b
\end{bmatrix} = [Ad_{T_{sb}}]\nu_b
$$

$$
[Ad_{T_{sb}}]^{-1}=[Ad_{T_{bs}}]=\begin{bmatrix}
R_{sb}^T & 0 \\
-R_{sb}^T[p_{sb}] & R_{sb}^T
\end{bmatrix}
$$

#### The Screw Interpretation of a Twist
A twist $\mathcal{V}$ can be interpreted in terms of a screw axis $\mathcal{S}$ and a velocity $\dot{\theta}$ about the screw axis. 
![](Attachements/Pasted%20image%2020250410194917.png)

A screw axis represents the familiar motion of a screw: rotating about the axis while also translating along the axis. One representation of a screw axis $\mathcal{S}$ is the collection ${ q, \hat{s}, h }$ , where $q \in \mathbb{R}^3$ is any point on the axis, $\hat{s}$ is a unit vector in the direction of the axis, and h is the screw pitch, which deﬁnes the ratio of the linear velocity along the screw axis to the angular velocity $\dot{\theta}$ about the screw axis.

Using the figure above and geometry

$$
\nu = \begin{bmatrix}
\omega \\
v
\end{bmatrix} = \begin{bmatrix}
\hat{s} \\
-\hat{s}\times q + h\hat{s}
\end{bmatrix} \dot{\theta} = \mathcal{S}\dot{\theta}
$$


$\mathcal{S_b}$ and $\mathcal{S_s}$ relation:

$$
\mathcal{S_s} = [Ad_{T_{sb}}]\mathcal{S_b}
$$

#### Exponential Coordinate Representation of Rigid-Body Motions
The Chasles–Mozzi theorem states that every rigid-body displacement can be expressed as a displacement along a ﬁxed screw axis S in space. 

By analogy to the exponential coordinates $\hat{\omega}\theta$ for rotations, we deﬁne the six-dimensional exponential coordinates of a homogeneous transformation $T$ as $\mathcal{S}\theta \in \mathbb{R}^6$ , where $\mathcal{S}$ is the screw axis and $\theta$ is the distance that must be traveled along the screw axis to take a frame from the origin $I$ to $T$. If the pitch of the screw axis is ﬁnite then $|\omega| = 1$ and $\theta$ corresponds to the angle of rotation about the screw axis. If the pitch of the screw is inﬁnite then $ω = 0$ and $|v| = 1$ and $θ$ corresponds to the linear distance traveled along the screw axis.

![](Attachements/Pasted%20image%2020250416165758.png)

**Exponential of $\mathcal{S}\theta$**

$$
e^{[\mathcal{S}]\theta}=\begin{bmatrix}
R & (I\theta+(1-cos(\theta))[\omega]+(\theta-sin(\theta))[\omega]^2)v \\
0 & 1
\end{bmatrix} = T
$$

#### Wrenches

The force which is shown in figure below, creates a torque or moment $m_a$ in the { a } frame:

$$
m_a = r_a \times f_a
$$

![](Attachements/Pasted%20image%2020250416170519.png)
Just as with twists, we can merge the moment and force into a single six-dimensional s**patial force**, or **wrench**, expressed in the { a } frame, $\mathcal{F}_a$ :

$$
\mathcal{F}_a = \begin{bmatrix}
m_a \\
f_a
\end{bmatrix}
$$

If more than one wrench acts on a rigid body, the total wrench on the body is simply the vector sum of the individual wrenches, provided that the wrenches are expressed in the same frame. 

**Relationship between $\mathcal{F}_a$ and $\mathcal{F}_b$:**
Recall that the dot product of a force and a velocity is a power, and power is a coordinate-independent quantity. Because of this, we know that:

$$
\nu^T_b\mathcal{F}_b = \nu^T_a\mathcal{F}_a=([Ad_{T_{ba}}]\nu_b)^T\mathcal{F}_a=\nu^T_b[Ad_{T_{ab}}]^T\mathcal{F}_a 
$$

$$
\implies \mathcal{F}_b = [Ad_{T_{ab}}]^T\mathcal{F}_a,~~~~~~ \mathcal{F}_a = [Ad_{T_{ba}}]^T\mathcal{F}_b
$$

### [Software](https://hades.mech.northwestern.edu/index.php/Modern_Robotics#Software)
- Inverse of a rotation matrix: RotInv(R)
- Skew-symmetric form of a vector: VecToso3(omg)
- Skew-symmetric to vector: So3ToVec(OMG)
- AxisAngle decomposition: \[omg, theta\] = AxisAng3(omg_theta)
- Skew-symmetric matrix to rotation matrix: MatrixExp3(skew_omg_theta)
- Rotation matrix to skew-symmetric form of omega: MatrixLog3(R)
- Creating T from R and p: T = RpToTrans(R,p)
- T to R and p: \[R, p\] = TransToRp(T)
- invT = TransInv(T)
- Creating se(3) matrix corresponding to a 6-vector twist V: se3mat = VecTose3(V)
- V = se3ToVec(se3mat)
- AdT = Adjoint(T)
- S = ScrewToAxis(q,s,h)
- \[S, theta\] = AxisAng6(S_theta)
- T = MatrixExp6(se3mat)
- se3mat = MatrixLog6(T)
---

## Robot Kinematics

### Forward Kinematics
The forward kinematics of a robot refers to the calculation of the position and orientation of its end-eﬀector frame from its joint coordinates θ. In this chapter we consider the forward kinematics of general open chains. One widely used representation for the forward kinematics of open chains relies on the **Denavit–Hartenberg parameters (D–H parameters)**. Another representation relies on the **product of exponentials (PoE)** formula.

#### Product of Exponentials Formula
To use the PoE formula, it is only necessary to assign a stationary frame { s } and a frame { b } at the end-eﬀector, described by $M$ when the robot is at its zero position. and a frame { b } at the end-eﬀector, described by $M$ when the robot is at its zero position. It is common to deﬁne a frame at each link, though, typically at the joint axis; these are needed for the D–H representation and they are useful for displaying a graphic rendering of a geometric model of the robot and for deﬁning the mass properties of the link.

##### First Formulation: Screw Axes in the Base Frame
The key concept behind the PoE formula is to regard each joint as applying a screw motion to all the outward links. To illustrate this consider a general spatial open chain like the one shown in figure below, consisting of n one-dof joints that are connected serially. To apply the PoE formula:

1. Choose a ﬁxed base frame { s } and an end-eﬀector frame { b } attached to the last link
2. Place the robot in its zero position by setting all joint values to zero, with the direction of positive displacement (rotation for revolute joints, translation for prismatic joints) for each joint speciﬁed. Let M ∈ SE(3) denote the conﬁguration of the end-eﬀector frame relative to the ﬁxed base frame when the robot is in its zero position.
3. Now suppose that joint n is displaced to some joint value θ n . The end-eﬀector frame M then undergoes a displacement of the form

$$
T = e^{[S_n]\theta_n}M
$$

4. Continuing with this reasoning and now allowing all the joints (θ 1 , . . . , θ n ) to vary, it follows that

$$
T = e^{[S_1]\theta_1}e^{[S_2]\theta_2} … e^{[S_{n-1}]\theta_{n-1}}e^{[S_n]\theta_n}M
$$

![](Attachements/Pasted%20image%2020250425161532.png)



**Example**

![](Attachements/Pasted%20image%2020250425162331.png)

$$M = \begin{bmatrix}
0 & 0 & -1 & L_1 \\
0 & 1 & 0 & 0 \\
-1 & 0 & 0 & L_2 \\
0 & 0 & 0 & 1
\end{bmatrix}$$
$$S_1 = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0\end{bmatrix}^T$$
$$S_2 = \begin{bmatrix}  0 & -1 & 0 & 0 & 0 & -L_1\end{bmatrix}^T$$
$$S_3 = \begin{bmatrix} 1 & 0 & 0 & 0 & -L_2 & 0 \end{bmatrix}^T$$
$$
T = e^{[S_1]\theta_1}e^{[S_2]\theta_2}e^{[S_3]\theta_3}M
$$

**Example**
![](Attachements/Pasted%20image%2020250425170307.png)

$$M = \begin{bmatrix}
-1 & 0 & 0 & L_1+L_2 \\
0 & 0 & 1 & W_1+W_2 \\
0 & 1 & 0 & H_1-H_2 \\
0 & 0 & 0 & 1
\end{bmatrix}$$

$$S_1 = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 \end{bmatrix}^T$$

$$S_2 = \begin{bmatrix} 0 & 1 & 0 & -H_1 & 0 & 0 \end{bmatrix}^T$$

$$S_3 = \begin{bmatrix} 0 & 1 & 0 & -H_1 & 0 & L_1 \end{bmatrix}^T$$

$$S_4 = \begin{bmatrix} 0 & 1 & 0 & -H_1 & 0 & -(L_1+L_2) \end{bmatrix}^T$$

$$S_5 = \begin{bmatrix} 0 & 0 & -1 &—W_1 & (L_1+L_2) & 0 \end{bmatrix}^T$$

$$S_6 = \begin{bmatrix} 0 & 1 & 0 & -(H_1-H_2) & 0 & L_1+L_2 \end{bmatrix}^T$$

##### Second Formulation: Screw Axes in the End-Eﬀector Frame

$$
e^{A^{-1}PA}=A^{-1}e^PA ~~~ \implies ~~~ Ae^{A^{-1}PA}=e^PA ~~~ \implies ~~~ Me^{M^{-1}PM} = e^PM
$$

Which means we can rewrite the transformation matrix as follows:

$$
T = Me^{M^{-1}S_1M}… e^{M^{-1}S_{n-1}M}e^{M^{-1}S_nM} = Me^{\mathcal{B}_1} … e^{\mathcal{B}_{n-1}}e^{\mathcal{B}_n}
$$

We call Equation above *the body form of the product of exponentials formula*. In the body form, $M$ is ﬁrst transformed by the ﬁrst joint, progressively moving outward to more distal joints. 

**Example**

![](Attachements/Pasted%20image%2020250425184130.png)

$$
M = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & L_1+L_2+L_3 \\
0 & 0 & 0 & 1 
\end{bmatrix}
$$
$$
\mathcal{B}_7 = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 \end{bmatrix}
$$
$$
\mathcal{B}_6 = \begin{bmatrix} 0 & 1 & 0 & L_3 & 0 & 0 \end{bmatrix}
$$
$$
\mathcal{B}_5 = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 \end{bmatrix}
$$
$$
\mathcal{B}_4 = \begin{bmatrix} 0 & 1 & 0 & L_2+L_3 & 0 & W_1 \end{bmatrix}
$$
$$
\mathcal{B}_3 = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 \end{bmatrix}
$$
$$
\mathcal{B}_2 = \begin{bmatrix} 0 & 1 & 0 & L_1+L_2+L_3 & 0 & 0 \end{bmatrix}
$$
$$
\mathcal{B}_1 = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 \end{bmatrix}
$$

#### Software
Software functions associated with this chapter are listed in MATLAB format below.

```
T = FKinBody(M,Blist,thetalist)
``` 
Computes the end-eﬀector frame given the zero position of the end-eﬀector M, the list of joint screws Blist expressed in the end-eﬀector frame, and the list of joint values thetalist.

```
T = FKinSpace(M,Slist,thetalist) 
```
Computes the end-eﬀector frame given the zero position of the end-eﬀector M, the list of joint screws Slist expressed in the ﬁxed-space frame, and the list of joint values thetalist.

### Velocity Kinematics and Statics
let us consider the case where the end-eﬀector conﬁguration is represented by a minimal set of coordinates $x ∈ \mathbb{R}^m$ and the velocity is given by $\dot{x} = \frac{dx}{dt} ∈ \mathbb{R}^m$ . In this case, the forward kinematics can be written as

$$
x(t) = f(\theta(t))
$$

where $\theta ∈ \mathbb{R}^n$ is a set of joint variables. By the chain rule, the time derivative at time t is

$$
\dot{x} = \frac{\partial f}{\partial \theta}\frac{d\theta}{dt} = J(\theta)\dot{\theta}
$$

where $J(θ) ∈ R^{m×n}$ is called the **Jacobian**. The Jacobian matrix represents the linear sensitivity of the end-eﬀector velocity $\dot x$ to the joint velocity $\dot θ$, and it is a function of the joint variables θ.
Usages of Jacobian matrix:
- To identify singularities: where $rank(J(\theta)) < max(rank(J(\theta)))$
- Manipulability ellipsoid: A map from max joints velocities to end-effector velocity
- Force ellipsoid: A map from max joints forces/moments to end-effector force/moment

**Note:** At a singularity, the manipulability ellipsoid collapses to a line segment. The force ellipsoid, on the other hand, becomes inﬁnitely long in a direction orthogonal to the manipulability ellipsoid line segment and skinny in the orthogonal direction.

![](Attachements/Pasted%20image%2020250611114859.png)

#### Manipulator Jacobian
The two standard types of Jacobian that we will consider are: the space Jacobian $J_s(θ)$ satisfying $V_s = J_s(θ) \dot θ$ where each column $J_{si} (θ)$ corresponds to a screw axis expressed in the ﬁxed space frame { s } ; and the body Jacobian $J_ b (θ)$ satisfying $V_ b = J_ b (θ) \dot θ$ where each column $J_ {bi} (θ)$ corresponds to a screw axis expressed in the end-eﬀector frame { b } .

##### Space Jacobian
Consider an n-link open chain whose forward kinematics is expressed in the following product of exponentials form:

$$
T(\theta_1, ... , \theta_n) = e^{[S_1]\theta_1}...e^{[S_n]\theta_n}M
$$

The spatial twist $\mathcal{V}_s$ is given by $[\mathcal{V}_s] = \dot T T^{-1}$, where

$$
\dot T = [S_1]e^{[S_1]\theta_1}\dot \theta_1 e^{[S_2]\theta_2} ... M + ... + e^{[S_1]\theta_1}...[S_n]e^{[S_n]\theta_n}\dot \theta_n
$$

$$
\dot T T^{-1} = [S_1]\theta_1 + ... + e^{[S_1]\theta_1}...e^{[S_{n-1}]\theta_{n-1}} [S_{n}] e^{-[S_1]\theta_1}...e^{-[S_{n-1}]\theta_{n-1}} \dot\theta_n
$$

$$
\implies \mathcal{V}_s = \begin{bmatrix}
J_{s1} & J_{s2}(\theta_1) & ... & J_{s_n}(\theta_1,...,\theta_{n-1})
\end{bmatrix} \vec{\dot\theta}
$$
Where
- $J_{s1} = S_1$
- $J_{si} = Ad_{e^{[S_1]\theta_1}...e^{S_{i-1}\theta_{i-1}}}S_i, ~~~~~~~ i \ge 2$ 

**Example**
![](Attachements/Pasted%20image%2020250428191534.png)

To write $J_s$ by hand all you need to do is to the S-List for all possible vector $\theta$s. Just write the rotation vector and linear component considering the lengths and orientations can be changed. And the other way is to use matrix exponential. 

$J_{s1} = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 \end{bmatrix}^T$
$J_{s2} = \begin{bmatrix} 0 & 0 & 1 & L_1s\theta_1 & -L_1c\theta_1 & 0 \end{bmatrix}^T$
$J_{s3} = \begin{bmatrix} 0 & 0 & 1 & L_1s\theta_1+L_2s(\theta_1+\theta_2) & -L_1c\theta_1-L_2c(\theta_1+\theta_2) & 0 \end{bmatrix}^T$
$J_{s4} = \begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}^T$

##### Body Jacobian
The process is the same as the Space Jacobian but inverse.

$$
\mathcal{V}_b = \mathcal{B}_n\dot\theta_n + ... + Ad_{e^{-\mathcal{B}_n}e^{-\mathcal{B}_{n-1}}...e^{-\mathcal{B}_2}}\dot\theta_1 = J_{bn}\dot\theta_n+...+J_{b1}\dot\theta_1
$$

##### Relationship between the Space and Body Jacobian

$$
J_s = Ad_{T_{sb}}J_b
$$

$$
\mathcal{V}_s = Ad_{T_{sb}}\mathcal{V}_b
$$

#### Statics of Open Chains
Using our familiar principle of conservation of power, we have power at the joints = (power to move the robot) + (power at the end-eﬀector) and, considering the robot to be at static equilibrium (no power is being used to move the robot), we can equate the power at the joints to the power at the end-eﬀector,

$$
\tau^T \dot\theta = \mathcal{F}^T\mathcal{V} ~~~~ \implies ~~~~ 
\tau^T \dot\theta = \mathcal{F}^TJ\dot\theta ~~~~ \implies ~~~~ 
\tau = J^T\mathcal{F}
$$

#### Singularity Analysis
The Jacobian allows us to identify postures at which the robot’s end-eﬀector loses the ability to move instantaneously in one or more directions. Such a posture is called a kinematic singularity, or simply a singularity. Mathematically, a singular posture is one in which the Jacobian J(θ) fails to be of maximal rank.

Kinematic singularities are also independent of the choice of ﬁxed frame and end-eﬀector frame. Choosing a diﬀerent ﬁxed frame is equivalent to simply relocating the robot arm, which should have absolutely no eﬀect on whether a particular posture is singular.

#### Manipulability
In the previous section we saw that, at a kinematic singularity, a robot’s endeﬀector loses the ability to translate or rotate in one or more directions. A kinematic singularity presents a binary proposition – a particular conﬁguration is either kinematically singular or it is not – and it is reasonable to ask if a nonsingular conﬁguration is “close” to being singular. The answer is yes; in fact, one can even determine the directions in which the end-eﬀector’s ability to move is diminished, and to what extent. The manipulability ellipsoid allows one to visualize geometrically the directions in which the end-eﬀector moves with least eﬀort or with greatest eﬀort.

Manipulability ellipsoids are illustrated for a 2R planar arm in the figure below.

![](Attachements/Pasted%20image%2020250430080043.png)

For a general n-joint open chain and a task space with coordinates $q ∈ R^m$ , where m ≤ n, the manipulability ellipsoid corresponds to the end-eﬀector velocities for joint rates $\dot θ$ satisfying $∥θ∥ = 1$, a unit sphere in the n-dimensional joint-velocity space. Assuming J is invertible, the unit joint-velocity condition can be written,

$$
1 = \dot \theta^T \dot \theta 
= (J^{-1}\dot q)^T(J^{-1}\dot q)
= \dot q^TJ^{-T}J^{-1}\dot q
= \dot q^T(JJ^{T})^{-1}\dot q = \dot q^TA^{-1}\dot q
$$

Letting $v_i$ and $λ_i$ be the eigenvectors and eigenvalues of $A=JJ^T$, the directions of the principal axes of the ellipsoid are $v_i$ and the lengths of the principal semi-axes are $\sqrt{λ_i}$ , as illustrated in the figure below (LinearAlgebra).

![](Attachements/Pasted%20image%2020250430081508.png)

When calculating the linear-velocity manipulability ellipsoid, it generally makes more sense to use the body Jacobian $J_b$ instead of the space Jacobian $J_s$ , since we are usually interested in the linear velocity of a point at the origin of the end-eﬀector frame rather than that of a point at the origin of the ﬁxed-space frame.

Apart from the geometry of the manipulability ellipsoid, it can be useful to assign a single scalar measure deﬁning how easily the robot can move at a given posture. One measure is the ratio of the longest and shortest semi-axes of the manipulability ellipsoid,

$$
\mu_1(A) = \sqrt\frac{\lambda_{max}(A)}{\lambda_{min}(A)}
$$

Some other measures:

$$
\mu_2(A) = \frac{\lambda_{max}(A)}{\lambda_{min}(A)}
$$

$$
\mu_3(A) = \sqrt{\lambda_{1}(A)\lambda_{2}(A)...\lambda_{n}(A)} = \sqrt{\det(A)}
$$

Just like the manipulability ellipsoid, a force ellipsoid can be drawn for joint torques τ satisfying $∥ τ ∥ = 1$. For the force ellipsoid, the matrix $B = JJ^T=A^{-1}$ plays the same role as A in the manipulability ellipsoid; it is the eigenvectors and the square roots of eigenvalues of B that deﬁne the shape of the force ellipsoid.

It's also conventional to analyse the jacobian matrix for linear motion and angular velocity seperatly 

$$
J = \begin{bmatrix} J_\omega \\ J_v \end{bmatrix} ~~~ \implies ~~~ A_\omega=J_\omega J_\omega^T, ~~ A_v=J_vJ_v^T
$$




#### Software
Software functions associated with this chapter are listed below.
- **Jb = JacobianBody(Blist, thetalist)** Computes the body Jacobian $J_b(θ) ∈ R^{6×n}$ given a list of joint screws $B_i$ expressed in the body frame and a list of joint angles.

- **Js = JacobianSpace(Slist, thetalist)** Computes the space Jacobian $J_s(θ) ∈ R^{6×n}$ given a list of joint screws $S_i$ expressed in the ﬁxed space frame and a list of joint angles.

### Inverse Kinematics


## Robot Dynamics

## Robot Motion Planning and Control

## Robot Manipulation and Wheeled Mobile Robots

## Capstone Project

## References
- _Modern Robotics: Mechanics, Planning, and Control_ , Kevin M. Lynch and Frank C. Park, Cambridge University Press 2017.