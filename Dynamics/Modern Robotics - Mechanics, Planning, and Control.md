
# Foundations of Robot Motion

## Configuration Space
he configuration of a robot is a complete specification of  the position of every point of the robot. The minimum number n of real-valued coordinates needed to represent the configuration is the number of degrees of freedom (dof) of the robot. The n-dimensional space containing all possible  
configurations of the robot is called the configuration space (C-space). The configuration of a robot is represented by a point in its C-space.

### Degrees of Freedom of a Rigid Body

$$
Dof = \text{Sum of freedoms of the pints on the object} ~ - ~ \text{Sum of Independent constraints}
$$
Since our robots consist of rigid bodies, Equation above can be expressed as follows:

$$
Dof = \text{Sum of freedoms of the bodies} ~ - ~ \text{Sum of Independent constraints}
$$

### Grubler’s Formula
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

### Some Notes

- A robot’s C-space can be parametrized explicitly or represented implicitly. For a robot with $n$ degrees of freedom, an explicit parametrization uses $n$ coordinates, the minimum necessary. An implicit representation involves m coordinates with $m \ge n$, with the m coordinates subject to $m - n$ constraint equations. With an implicit parametrization, a robot’s C-space can be viewed as a surface of dimension n embedded in a space of higher dimension m.

- A robot’s task space is a space in which the robot’s task can be naturally expressed. A robot’s workspace is a specification of the configurations that the end-effector of the robot can reach.
# Robot Kinematics

# Robot Dynamics

# Robot Motion Planning and Control

# Robot Manipulation and Wheeled Mobile Robots

# Capstone Project

# References
- _Modern Robotics: Mechanics, Planning, and Control_ , Kevin M. Lynch and Frank C. Park, Cambridge University Press 2017.