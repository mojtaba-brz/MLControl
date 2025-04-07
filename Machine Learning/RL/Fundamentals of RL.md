# Fundamentals of Reinforcement Learning

- [[#Introduction|Introduction]]
	- [[#Introduction#Reinforcement Learning|Reinforcement Learning]]
	- [[#Introduction#Elements of Reinforcement Learning|Elements of Reinforcement Learning]]
- [[#Multi-armed Bandits (A review on RL concepts)|Multi-armed Bandits (A review on RL concepts)]]
	- [[#Multi-armed Bandits (A review on RL concepts)#Some basic solution to exploration-exploitation dillema|Some basic solution to exploration-exploitation dillema]]
- [[#How to do Real World RL|How to do Real World RL]]
	- [[#How to do Real World RL#Priorities|Priorities]]
- [[#Core Ideas Of Reinforcement Learning Algorithms|Core Ideas Of Reinforcement Learning Algorithms]]
	- [[#Core Ideas Of Reinforcement Learning Algorithms#Markov Decision Process|Markov Decision Process]]
	- [[#Core Ideas Of Reinforcement Learning Algorithms#Goals and Rewards|Goals and Rewards]]
	- [[#Core Ideas Of Reinforcement Learning Algorithms#Returns and Episodes|Returns and Episodes]]
	- [[#Core Ideas Of Reinforcement Learning Algorithms#Policies and Value Functions|Policies and Value Functions]]
	- [[#Core Ideas Of Reinforcement Learning Algorithms#Optimal Policies and Optimal Value Functions|Optimal Policies and Optimal Value Functions]]
- [[#Dynamic Programming|Dynamic Programming]]
	- [[#Dynamic Programming#Policy Evaluation (Prediction)|Policy Evaluation (Prediction)]]
	- [[#Dynamic Programming#Policy Improvement|Policy Improvement]]
	- [[#Dynamic Programming#Policy Iteration|Policy Iteration]]
	- [[#Dynamic Programming#Generalized Policy Iteration|Generalized Policy Iteration]]
	- [[#Dynamic Programming#Efficiency of Dynamic Programming|Efficiency of Dynamic Programming]]
- [[#Monte Carlo Methods|Monte Carlo Methods]]
	- [[#Monte Carlo Methods#Monte Carlo Prediction|Monte Carlo Prediction]]
	- [[#Monte Carlo Methods#Monte Carlo Estimation of Action Values|Monte Carlo Estimation of Action Values]]
	- [[#Monte Carlo Methods#Monte Carlo Control|Monte Carlo Control]]
	- [[#Monte Carlo Methods#Monte Carlo Control without Exploring Starts|Monte Carlo Control without Exploring Starts]]
		- [[#Monte Carlo Control without Exploring Starts#On-Policy MC Control|On-Policy MC Control]]
		- [[#Monte Carlo Control without Exploring Starts#Off-Policy MC Control via Importance Sampling|Off-Policy MC Control via Importance Sampling]]
- [[#Temporal-Difference Learning|Temporal-Difference Learning]]
	- [[#Temporal-Difference Learning#TD Prediction|TD Prediction]]
	- [[#Temporal-Difference Learning#Advantages of TD Prediction Methods|Advantages of TD Prediction Methods]]
	- [[#Temporal-Difference Learning#Sarsa: On-policy TD Control|Sarsa: On-policy TD Control]]
	- [[#Temporal-Difference Learning#Q-learning: Off-policy TD Control|Q-learning: Off-policy TD Control]]
	- [[#Temporal-Difference Learning#Expected Sarsa|Expected Sarsa]]
	- [[#Temporal-Difference Learning#Maximization Bias and Double Learning|Maximization Bias and Double Learning]]
	- [[#Temporal-Difference Learning#Open challenges|Open challenges]]
- [[#$n$-step Bootstrapping|$n$-step Bootstrapping]]
- [[#References|References]]


## Introduction
The idea that we learn by interacting with our environment is probably the first to occur to us when we think about the nature of learning. When an infant plays, waves its arms, or looks about, it has no explicit teacher, but it does have a direct sensorimotor connection to its environment. Exercising this connection produces a wealth of information about cause and effect, about the consequences of actions, and about what to do in order to achieve goals. Throughout our lives, such interactions are undoubtedly a major source of knowledge about our environment and ourselves. Whether we are learning to drive a car or to hold a  onversation, we are acutely aware of how our environment responds to what we do, and we seek to influence what happens through our behavior. Learning from interaction is a foundational idea underlying nearly all theories of learning and intelligence.

The approach we explore here, called reinforcement learning, is much more focused on goal-directed learning from interaction than are other  approaches to machine learning.

### Reinforcement Learning
**Reinforcement learning is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal.** 

### Elements of Reinforcement Learning
1. **Agent**
2. **Environment**
3. **Policy** : Roughly speaking, a policy is a mapping from perceived states of the environment to actions to be taken when in those states. In some cases the policy may be a simple function or lookup table, whereas in others it may involve extensive computation such as a search process. In general, policies may be stochastic, specifying probabilities for each action.
4. **Reward signal** : The agent’s sole objective is to maximize the total reward it receives over the long run. The reward signal thus defines what are the good and bad events for the agent. In a biological system, we might think of rewards as analogous to the experiences of pleasure or pain. They are the immediate and defining features of the problem faced by the agent. The reward signal is the primary basis for altering the policy; if an action selected by the policy is followed by low reward, then the policy may be changed to select some other action in that situation in the future. In general, reward signals may be stochastic functions of the state of the environment and the actions taken.
5. **Value function** : Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Whereas rewards determine the immediate, intrinsic desirability of environmental states, values indicate the long-term desirability of states after taking into account the states that are likely to follow and the rewards available in those states.In fact, the most important component of almost all reinforcement learning algorithms we consider is a method for efficiently estimating values.
6. **Model of the environment (optional)**

## Multi-armed Bandits (A review on RL concepts)
Consider the following learning problem. You are faced repeatedly with a choice among k different options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. Your objective is to maximize the expected total reward over some time period, for example, over 1000 action selections, or *time steps*. 

our k-armed bandit problem, each of the k actions has an **expected or mean reward** given that that action is selected; let us call this the value of that action. If you knew the value of each action, then it would be trivial to solve the k-armed bandit problem: you would always select the action with highest value. If you maintain estimates of the action values, then at any time step there is at least one action whose estimated value is greatest. We call these the greedy actions. When you select one of these actions, we say that you are **exploiting** your current knowledge of the values of the actions. If instead you select one of the non-greedy actions, then we say you are **exploring**, because this enables you to improve your estimate of the non-greedy action’s value.

For more informations on this subject, you can refer to RL:An introduction book by Sutton an Burto.

### Some basic solution to exploration-exploitation dillema
1. $\epsilon-greedy$ method
	1. Choose the best action with 1-$\epsilon$ probability
	2. Choose a random action with $\epsilon$ probability
2. Optimistic initial values
	1. Initialise the action-values or state values high enough to make sure the agent is going to try them all enough at the beginning 
3. Upper confidence boundary (UCB)

## How to do Real World RL
The answer is, by shifting the learning process’ priorities. 
### Priorities 
• Generalisation over temporal credit assignments
• Considering that the environment controls not the agent
• Statistical efficiency over computational efficiency
• You need to thinks about features not the states
• Evaluation over Learning
• Every policy you are implementing in the real world is important but in the simulator only the last policy is important

## Core Ideas Of Reinforcement Learning Algorithms
### Markov Decision Process
MDPs are a mathematically idealized form of the reinforcement learning problem for which precise theoretical statements can be made. 
![RL MDPs](./Attachements/MDP.png)

The function $p$ defines the dynamic of the MDP:

$$
p(s', r | s, a) = Pr[S_{t+1} = s', R_{t+1} = r | S_{t} = s, A_{t} = a]
$$

$S \text{: States}$

$R \text{: Reward}$

$A \text{: Action}$

From the four-argument dynamics function, p, one can compute anything else one might want to know about the environment, such as the *state-transition probabilities*,

$$
p(s'| s, r) = \sum_{r} p(s', r | s, a)
$$

$$
r(s, a) = E[R_t|S_{t} = s, A_{t} = a] = \sum_{r} r \sum_{s'} p(s', r | s, a)
$$

$$
r(s', a, s) = E[R_t|S_{t+1} = s', S_{t} = s, A_{t} = a] = \sum_{r} r \space \frac{p(s', r | s, a)}{p(s'| s, r)}
$$

### Goals and Rewards
In reinforcement learning, the purpose or goal of the agent is formalized in terms of a special signal, called the reward, passing from the environment to the agent. Informally, the agent’s goal is to maximize the total amount of reward it receives. This means maximizing not immediate reward, but cumulative reward in the long run. We can clearly state this informal idea as the reward hypothesis:

**That all of what we mean by goals and purposes can be well thought of as
the maximization of the expected value of the cumulative sum of a received
scalar signal (called reward).**

The reward signal is your way of communicating to the robot what you want it to achieve, not how you want it achieved.

### Returns and Episodes
In general, we seek to maximize the expected return, where the return, denoted $G_t$, is defined as some specific function of the reward sequence.

$$
G_t = \sum_{k=1}^{T} \gamma^{k-1} R_{t+k} = R_{t+1} + \gamma G_{t+1}
$$

$T \text{: Final time step}$

$\gamma \text{: Discount rate, } \space \space \space \space 0\le \gamma\le 1$

Each episode ends in a special
state called the terminal state, followed by a reset to a standard starting state or to a
sample from a standard distribution of starting states. Even if you think of episodes as ending in differerent ways, such as winning and losing a game, the next episode begins independently of how the previous one ended. Thus the episodes can all be considered to end in the same terminal state, with differerent rewards for the differerent outcomes. Tasks with episodes of this kind are called episodic tasks.

On the other hand, in many cases the agent–environment interaction does not break naturally into identifiable episodes, but goes on continually without limit. For example, this would be the natural way to formulate an on-going process-control task, or an application to a robot with a long life span. We call these continuing tasks.

### Policies and Value Functions
Almost all reinforcement learning algorithms involve estimating value functions—functions of states (or of state–action pairs) that estimate how good it is for the agent to be in a given state (or how good it is to perform a given action in a given state). The notion of “how good” here is defined in terms of future rewards that can be expected, or, to be precise, in terms of expected return. Of course the rewards the agent can expect to receive in the future depend on what actions it will take. Accordingly, value functions are defined with respect to particular ways of acting, called policies.

Formally, a policy is a mapping from states to probabilities of selecting each possible action. If the agent is following policy $\pi$ at time t, then $\pi(a|s)$ is the probability that $A_t = a$ if $S_t = s$.

The value function of a state $s$ under a policy $\pi$, denoted $v_\pi(s)$, is the expected return
when starting in s and following $\pi$ thereafter. For MDPs, we can define $v_\pi$ formally by

$$
v_\pi(s) = E[G_t|S_t = s] 
$$

$$
\implies v_\pi(s) = E[\sum_{k=1}^{T} \gamma^{k-1} R_{t+k}|S_t = s] 
$$

$$
\implies \boxed{v_\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a)(r + \gamma v_{\pi}(s'))}
$$

Similarly, we define the value of taking action a in state s under a policy $\pi$, denoted $q_\pi(s, a)$ ( action-value function), as the expected return starting from s, taking the action a, and thereafter
following policy $\pi$:

$$
q_\pi(s, a) = E[G_t|S_t = s, A_t = a] 
$$

$$
\implies q_\pi(s, a) = E[\sum_{k=1}^{T} \gamma^{k-1} R_{t+k}|S_t = s, A_t = a] 
$$

$$
\implies \boxed{q_\pi(s, a) = \sum_{s', r} p(s', r | s, a)(r + \gamma v_{\pi}(s'))}
$$

### Optimal Policies and Optimal Value Functions
A policy $\pi$ is defined to be better than or equal to a policy $\pi'$ if its expected return is greater than or equal to that of $\pi'$ for all states. In other words, $\pi$ > $\pi'$ if and only if $v_\pi$(s) > $v_{\pi'}$(s) for all $s \in S$. There is always at least one policy that is better than or equal to all other policies. This is an optimal policy. Although there may be more than one, we denote all the optimal policies by $\pi*$. They share the same state-value function, called the optimal state-value function, denoted $v_*$, and defined as

$$
v_{*}(s) = \max_{\pi} v_{\pi}(s) 
$$

$$
\implies v_{\star}(s) = \max_{a} \sum_{s', r} p(s', r \mid s, a)(r + \gamma v_{\star}(s'))
$$

Optimal policies also share the same optimal action-value function,

$$
q_{*}(s,a) = \max_{\pi} q_{\pi}(s,a) 
$$

$$
\implies q_{\star}(s,a) = \sum_{s', r} p(s', r \mid s, a)(r + \gamma \max_{a'}q_{\star}(s', a'))
$$

## Dynamic Programming
The term dynamic programming (DP) refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a Markov decision process (MDP). Classical DP algorithms are of limited utility in reinforcement learning both because of their assumption of a perfect model and because of their great computational expense, but they are still important theoretically. DP provides an essential foundation for the understanding of the methods presented in RL. In fact, all of these methods can be viewed as attempts to achieve much the same eect as DP, only with less computation and without assuming a perfect model of the environment.

The key idea of DP, and of reinforcement learning generally, is the use of value functions to organize and structure the search for good policies. We can easily obtain optimal policies once we have found the optimal value functions, $v_*$ or $q_*$, which satisfy the Bellman optimality equations.

### Policy Evaluation (Prediction)

First we consider how to compute the state-value function $v_\pi$ for an arbitrary policy $\pi$. This is called policy evaluation in the DP literature. We also refer to it as the prediction problem.

If the environment’s dynamics are completely known, then problem becomes a system of $|S|$ simultaneous linear equations in $|S|$ unknowns (the $v_\pi(s)$, $s \in S$). In principle, its solution is a straightforward, if tedious, computation. For our purposes, iterative solution methods are most suitable:

``` 
Inputs : 
	pi : the policy we want to evaluate,
	Theta : iterations stop threshold,
	Gamma : decaying factor of value-function

Algorithm :
	Initialise V for all states (S) and set V(terminal) to zero
	loop:
		delta = 0
		For s in S:
			v = V(s)
			V(s) = sum(pi(s, a) * sum(p(s_p, r, s, a) * (r + gamma*V(s_p))))
			Delta = max(delta, abs(v - V(s)))
		if Delta > theta:
			Break
```

### Policy Improvement
Our reason for computing the value function for a policy is to help ﬁnd better policies. The key idea is to improve the policy $\pi$ such that:

$$
v_{\pi’} \ge v_{\pi}
$$

At least in a state. One option is to consider $\pi’$ as:

$$
\pi’ = \max_a \sum_{s’ , r} p(s’, r|s, a)(r + \gamma v(s‘))
$$
Following this idea, policy improvement thus must give us a strictly better policy except when the original policy is already optimal.

### Policy Iteration
Once a policy, $\pi$, has been improved using $v_\pi$ to yield a better policy, $\pi’$ , we can then compute $v_{\pi’}$  and improve it again to yield an even better $\pi^”$ . This way of ﬁnding an optimal policy is called policy iteration. A complete algorithm is given in the box below. Note that each policy evaluation, itself an iterative computation, is started with the value function for the previous policy. This typically results in a great increase in the speed of convergence of policy evaluation (presumably because the value function changes little from one policy to the next).

```
Policy Iteration Algorithm:
	Initialise V for all states (S) and policy pi for all actions and states

	Loop:
		Run policy evaluation algorithm

		policy_stable = true
		for s in S:
			old_action = pi(s)
			pi(s) = argmax_a sum(p(s_p, r, s, a) * (r + gamma*V(s_p))))
	
			If old_action != pi(s):
				policy_stable = false
		
		if policy_stable:
			Break
```

### Generalized Policy Iteration
Policy iteration consists of two simultaneous, interacting processes, one making the value function consistent with the current policy (policy evaluation), and the other making the policy greedy with respect to the current value function (policy improvement). In policy iteration, these two processes alternate, each completing before the other begins, but this is not really necessary. In value iteration, for example, only a single iteration of policy evaluation is performed in between each policy improvement. In asynchronous DP methods, the evaluation and improvement processes are interleaved at an even ﬁner grain. In some cases a single state is updated in one process before returning to the other. As long as both processes continue to update all states, the ultimate result is typically the same—convergence to the optimal value function and an optimal policy.
We use the term generalized policy iteration (GPI) to refer to the general idea of letting policy-evaluation and policy- evaluation improvement processes interact, independent of the granularity and other details of the two processes. Almost all reinforcement learning methods are well described as GPI. That is, all have ⇡ V identiﬁable policies and value functions, with the policy always being improved with respect to the value function and the value ⇡ ! greedy(V ) function always being driven toward the value function for the improvement policy, as suggested by the diagram to the right. If both the evaluation process and the improvement process stabilize, that is, no longer produce changes, then the value function and policy must be optimal. The value function stabilizes only when it is consistent with the current policy, and the policy stabilizes only when it is greedy with respect to the current value function. Thus, both processes stabilize only when a policy has been found that is greedy with respect to its own evaluation function. This implies that the Bellman optimality equation holds, and thus that the policy and the value function are optimal.

### Efficiency of Dynamic Programming

DP may not be practical for very large problems, but compared with other methods for solving MDPs, DP methods are actually quite efficient. If n and k denote the number of states and actions, this means that a DP method takes a number of computational operations that is less than some polynomial function of n and k. A DP method is guaranteed to ﬁnd an optimal policy in polynomial time even though the total number of (deterministic) policies is $k^n$ . In this sense, DP is exponentially faster than any direct search in policy space could be, because direct search would have to exhaustively examine each policy to provide the same guarantee. Linear programming methods can also be used to solve MDPs, and in some cases their worst-case convergence guarantees are better than those of DP methods. But linear programming methods become impractical at a much smaller number of states than do DP methods (by a factor of about 100). For the largest problems, only DP methods are feasible.

DP is sometimes thought to be of limited applicability because of the curse of dimensionality, the fact that the number of states often grows exponentially with the number of state variables. Large state sets do create di!culties, but these are inherent di!culties of the problem, not of DP as a solution method. In fact, DP is comparatively better suited to handling large state spaces than competing methods such as direct search and linear programming.

On problems with large state spaces, asynchronous DP methods are often preferred. To complete even one sweep of a synchronous method requires computation and memory for every state. For some problems, even this much memory and computation is impractical, yet the problem is still potentially solvable because relatively few states occur along optimal solution trajectories. Asynchronous methods and other variations of GPI can be applied in such cases and may ﬁnd good or optimal policies much faster than synchronous methods can.

## Monte Carlo Methods
Monte Carlo methods require only experience—sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. Learning from actual experience is striking because it requires no prior knowledge of the environment’s dynamics, yet can still attain optimal behavior.

Monte Carlo methods are ways of solving the reinforcement learning problem based on averaging sample returns. To ensure that well-deﬁned returns are available, here we deﬁne Monte Carlo methods only for episodic tasks. The term “Monte Carlo” is often used more broadly for any estimation method whose operation involves a signiﬁcant random component. Here we use it speciﬁcally for methods based on averaging complete returns (as opposed to methods that learn from partial returns, considered in the next chapter).

Because all the action selections are undergoing learning, the problem becomes nonstationary from the point of view of the earlier state. To handle the nonstationarity, we adapt the idea of general policy iteration (GPI). 

### Monte Carlo Prediction
We begin by considering Monte Carlo methods for learning the state-value function for a given policy. In particular, suppose we wish to estimate $v_{\pi(s)}$d, the value of a state s under policy $\pi$, given a set of episodes obtained by following $\pi$ and passing through $s$. Each occurrence of state $s$ in an episode is called a visit to $s$. Of course, $s$ may be visited multiple times in the same episode; let us call the ﬁrst time it is visited in an episode the ﬁrst visit to $s$. The ﬁrst-visit MC method estimates $v_{\pi(s)}$ as the average of the returns following ﬁrst visits to s, whereas the every-visit MC method averages the returns following all visits to $s$. These two Monte Carlo (MC) methods are very similar but have slightly different theoretical properties. First-visit MC has been most widely studied, dating back to the 1940s, and is the one we focus on in this chapter. Every-visit MC extends more naturally to function approximation and eligibility traces, as discussed in Chapters 9 and 12. First-visit MC is shown in procedural form in the box. Every-visit MC would be the same except without the check for $S_t$ having occurred earlier in the episode.

**First-visit MC prediction, for estimating $V_{\pi(s)}$**
```
Input : policy pi

Inititialzation:
V(s) for all s in S
Reterns(s) an empty list for all s in S

While True:
	Generate and episode following pi
	G = 0
	for t from T-1 to 0:
		G = gamma * G + R(t+1)
		if s(t) is not in s(t-1) to s(0):
			Returns(s(t)) += G
			V(s(t)) = average(Returns(s(t)))
```

### Monte Carlo Estimation of Action Values
If a model is not available, then it is particularly useful to estimate action values (the values of state–action pairs) rather than state values. The only complication is that many state–action pairs may never be visited. If $\pi$ is a deterministic policy, then in following $\pi$ one will observe returns only for one of the actions from each state. With no returns to average, the Monte Carlo estimates of the other actions will not improve with experience. This is a serious problem because the purpose of learning action values is to help in choosing among the actions available in each state. To compare alternatives we need to estimate the value of all the actions from each state, not just the one we currently favor.

This is the general problem of maintaining exploration. For policy evaluation to work for action values, we must assure continual exploration. One way to do this is by specifying that the episodes start in a state–action pair, and that every pair has a nonzero probability of being selected as the start. This guarantees that all state–action pairs will be visited an inﬁnite number of times in the limit of an inﬁnite number of episodes. We call this the assumption of exploring starts.

The assumption of exploring starts is sometimes useful, but of course it cannot be relied upon in general, particularly when learning directly from actual interaction with an environment. In that case the starting conditions are unlikely to be so helpful. The most common alternative approach to assuring that all state–action pairs are encountered is to consider only policies that are stochastic with a nonzero probability of selecting all actions in each state. We discuss two important variants of this approach in later sections. For now, we retain the assumption of exploring starts and complete the presentation of a full Monte Carlo control method.

### Monte Carlo Control
For Monte Carlo policy iteration it is natural to alternate between evaluation and improvement on an episode-by-episode basis. After each episode, the observed returns are used for policy evaluation, and then the policy is improved at all the states visited in the episode. A complete simple algorithm along these lines, which we call Monte Carlo ES:
**Monte Carlo ES for estimating the optimal policy $\pi_\star$
```
Initialisation:
pi(s) for all actions in all states
Q(s, a) Arbitrary for all s and a
Returns(s, a) an empty list for s and a

While True:
	Choose s(0) and a(0) randomly
	Generate an episode following pi and starting with s(0) and a(0)
	G = 0

	for t from T-1 to 0:
		G = gamma * G + R(t+1)

		if (s(t), a(t)) is not in (s(t-1), a(t-1)) to (s(0), a(0)):
			append G to Returns(s(t), a(t))
			Q(s(t), a(t)) = average(Returns(s(t), a(t))) 
			pi(s(t))      = argmax_a(Q(s(t), :))
 ```

### Monte Carlo Control without Exploring Starts
How can we avoid the unlikely assumption of exploring starts? How can we avoid the unlikely assumption of exploring starts? The only general way to ensure that all actions are selected inﬁnitely often is for the agent to continue to select them. There are two approaches to ensuring this, resulting in what we call on-policy methods and off-policy methods. On-policy methods attempt to evaluate or improve the policy that is used to make decisions, whereas off-policy methods evaluate or improve a policy different from that used to generate the data.

#### On-Policy MC Control
In on-policy control methods the policy is generally soft, meaning that $\pi(a | s) > 0$ for all $s \in S$ and all $a \in A(s)$, but gradually shifted closer and closer to a deterministic optimal policy. The on-policy method we present in this section uses $\epsilon-greedy$ policies, meaning that most of the time they choose an action that has maximal estimated action value, but with probability $\epsilon$ they instead select an action at random. The $\epsilon-greedy$ policies are examples of $\epsilon-soft$ policies. The complete algorithm for on-policy MC control is given in the box below:

**On-policy MC Control Algorithm:**
```
Parameters : epsilon, gamma
Initialisation:
pi : an arbitrary e-greedy policy
Q and Returns

While True:
	Generate an episode
	G = 0
	for t from T-1 to 0:
		G = gamma * G + R(t+1)
		
		if (s(t), a(t)) is not in (s(t-1), a(t-1)) to (s(0), a(0)):
			append G to Returns(s(t), a(t))
			Q(s(t), a(t)) = average(Returns(s(t), a(t))) 
			a_best        = argmax_a(Q(s(t), :))
			update pi(s(t),:) based on a_best

```

#### Off-Policy MC Control via Importance Sampling
All learning control methods face a dilemma: They seek to learn action values conditional on subsequent optimal behavior, but they need to behave non-optimally in order to explore all actions (to ﬁnd the optimal actions). How can they learn about the optimal policy while behaving according to an exploratory policy? A more straightforward approach is to use two policies, one that is learned about and that becomes the optimal policy, and one that is more exploratory and is used to generate behavior. The policy being learned about is called the target policy, and the policy used to generate behavior is called the behavior policy. In this case we say that learning is from data “off” the target policy, and the overall process is termed off-policy learning.

Off-policy methods require additional concepts and notation, and because the data is due to a different policy, off-policy methods are often of greater variance and are slower to converge. On the other hand, off-policy methods are more powerful and general. They include on-policy methods as the special case in which the target and behavior policies are the same. Off-policy methods also have a variety of additional uses in applications. For example, they can often be applied to learn from data generated by a conventional non-learning controller, or from a human expert.

In this section we begin the study of o↵-policy methods by considering the prediction problem, in which both target and behavior policies are ﬁxed. That is, suppose we wish to estimate $v_\pi$ or $q_\pi$ , but all we have are episodes following another policy $b$, where $b \ne \pi$. In this case, $\pi$ is the target policy, $b$ is the behavior policy, and both policies are considered ﬁxed and given.

In order to use episodes from b to estimate values for$\pi$, we require that every action taken under $\pi$ is also taken, at least occasionally, under $b$. That is, we require that $\pi(a | s) > 0 \implies b(a | s) > 0$. This is called the assumption of coverage. It follows from coverage that $b$ must be stochastic in states where it is not identical to $\pi$. The target policy $\pi$, on the other hand, may be deterministic, and, in fact, this is a case of particular interest in control applications. In control, the target policy is typically the deterministic greedy policy with respect to the current estimate of the action-value function. This policy becomes a deterministic optimal policy while the behaviour policy remains stochastic and more exploratory, for example, an $\epsilon-greedy$ policy. In this section, however, we consider the prediction problem, in which $\pi$ is unchanging and given.

Almost all off-policy methods utilise importance sampling, a general technique for estimating expected values under one distribution given samples from another. We apply importance sampling to off-policy learning by weighting returns according to the relative probability of their trajectories occurring under the target and behavior policies, called the *importance-sampling ratio*. Given a starting state $S_t$ , the probability of the subsequent state–action trajectory, $A_t$ , $S_{t+1} , A_{t+1} , . . . , S_T$ , occurring under any policy $\pi$ is

$$
\prod_{k=t}^{T-1} \pi_(a_k|s_k)p(s_{k+1}|s_k,a_k)
$$

Thus, the relative probability of the trajectory under the target and behavior policies (the importance sampling ratio) is

$$
\rho_{t:T-1} = \frac{\prod_{k=t}^{T-1} \pi_(a_k|s_k)p(s_{k+1}|s_k,a_k)}{\prod_{k=t}^{T-1} b_(a_k|s_k)p(s_{k+1}|s_k,a_k)} = \prod_{k=t}^{T-1}\frac{\pi_(a_k|s_k)}{b_(a_k|s_k)}
$$

Although the trajectory probabilities depend on the MDP’s transition probabilities, which are generally unknown, they appear identically in both the numerator and denominator, and thus cancel. The importance sampling ratio ends up depending only on the two policies and the sequence, not on the MDP. The ratio $\rho_{t:T-1}$ transforms the returns to have the right expected value:

$$
v_{\pi(s)} = E[\rho_{t:T-1}G_t~|~s = S_t]
$$

**Ordinary Importance Sampling:**
$$
v_{\pi(s)} \doteq \frac{\sum_{\text{all episodes}}\rho_{t:T-1}G_t~|~s = S_t}{\text{num of state s was visited in total}} ~~~~~~~~~~~ \text{every visit MC}
$$
**Note:** $v_{\pi(s)}$ is zero if the denominator is zero.

**Weighted Importance Sampling:**

$$
v_{\pi(s)} \doteq \frac{\sum_{\text{all episodes}}\rho_{t:T-1}G_t~|~s = S_t}{\sum_{\text{all episodes}}\rho_{t:T-1}}
$$

**Note:** $v_{\pi(s)}$ is zero if the denominator is zero.

Ordinary importance sampling is unbiased whereas weighted importance sampling is biased (though the bias converges asymptotically to zero). On the other hand, the variance of ordinary importance sampling is in general unbounded because the variance of the ratios can be unbounded, whereas in the weighted estimator the largest weight on any single return is one. In fact, assuming bounded returns, the variance of the weighted importance-sampling estimator converges to zero even if the variance of the ratios themselves is inﬁnite (Precup, Sutton, and Dasgupta 2001). In practice, the weighted estimator usually has dramatically lower variance and is strongly preferred. Nevertheless, we will not totally abandon ordinary importance sampling as it is easier to extend to the approximate methods using function approximation.

The every-visit methods for ordinary and weighed importance sampling are both biased, though, again, the bias falls asymptotically to zero as the number of samples increases. In practice, every-visit methods are often preferred because they remove the need to keep track of which states have been visited and because they are much easier to extend to approximations.

**Ordinary Importance Sampling Prediction using every visit MC Algorithm:**
```
Input : policy pi, b

Inititialzation:
V(s) for all s in S
Reterns(s) an empty list for all s in S

While True:
	Generate and episode following pi
	G = 0
	W = 1 // rho
	for t from T-1 to 0:
		G   = gamma * G + R(t+1)
		W   = W * pi(a(t), s(t)/b(a(t), s(t)
		Append W * G to Returns(s(t))
		V(s(t)) = average(Returns(s(t)))
```


## Temporal-Difference Learning
If one had to identify one idea as central and novel to reinforcement learning, it would  
undoubtedly be temporal-difference (TD) learning. Like Monte Carlo methods,  
TD methods can learn directly from raw experience without a model of the environment’s  
dynamics. Like DP, TD methods update estimates based in part on other learned  
estimates, without waiting for a final outcome (they bootstrap).

### TD Prediction
A simple every-visit Monte Carlo method suitable for nonstationary  
environments is

$$
V(s) = V(s) + \alpha(G_t - V(s))
$$

Whereas Monte Carlo methods must wait until the end of the episode, TD methods need to wait only until the next time step.
The simplest TD method makes the update:

$$
V(s) = V(s) + \alpha(r + \gamma V(s') - V(s))
$$

This TD method is called TD(0), or one-step TD. The box below specifies TD(0) completely in procedural form.

**Tabular TD(0) Prediction** 
```
Input : Policy pi to eval

Params : alpah, gamma

Initialization: V(s) for all s and zero to terminal state

while True:
	initialize s
	while s is not terminal:
		a = pi(s)
		r, s_next = env(a)
		V(s) += alpha * (r + gamma * V(s_next) - V(s))
		s = s_next
```

### Advantages of TD Prediction Methods
- Naturally implemented in an online, fully incremental fashion
- Do not require a model of the environment
- For any fixed policy $\pi$, TD(0) has been proved to converge to $v_\pi$, in the mean for a constant step-size parameter if it is sufficiently small

### Sarsa: On-policy TD Control
Update rule:
 
 $$
Q(s_t, a_t) += \alpha(r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)) 
 $$
To use this update rule we need to have $s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1}$ which can be said as SARSA.

**SARSA Algorithm:**
```
Inputs: an epsilon-greedy policy, pi that uses Q

Params : alpha, gamma

Initialization : q

while True:
	choose an s
	a = pi(s, q)
	while s in not Terminal:
		r, s_next = env(a)
		a_next = pi(s_next, q)
		q(s, a) += alpha * (r + gamma * q(s_next, a_next) - q(a, s))
		s, a = s_next, a_next
```

### Q-learning: Off-policy TD Control
Update rule:
 
 $$
Q(s_t, a_t) += \alpha(r_{t+1} + \gamma~ ~\max_a(Q(s_{t+1}), a) - Q(s_t, a_t)) 
 $$

Q-learning Algorithm:**
```
Inputs: an epsilon-greedy policy, pi that uses Q

Params : alpha, gamma

Initialization : q

while True:
	choose an s
	while s in not Terminal:
		a = pi(s, q)
		r, s_next = env(a)
		q(s, a) += alpha * (r + gamma * max(q(s_next, :)) - q(a, s))
		s = s_next
```

### Expected Sarsa
Update rule:
 
 $$
Q(s_t, a_t) += \alpha(r_{t+1} + \gamma~ ~\sum_a\pi(a|s_{t+1})(Q(s_{t+1}, a) - Q(s_t, a_t)) 
$$

Expected Sarsa is more complex computationally than Sarsa but, in return, it eliminates the variance due to the random selection of $A_t+1$. Given the same amount of experience we might expect it to perform slightly better than Sarsa, and indeed it generally does.

![](Attachements/Pasted%20image%2020250407213532.png)

in general Expected Sarsa might use a policy different from the target policy ⇡ to generate behavior, in which case it becomes an off-policy algorithm. For example, suppose $\pi$ is the greedy policy while behavior is more exploratory; then Expected Sarsa is exactly Q-learning. In this sense  
Expected Sarsa subsumes and generalizes Q-learning while reliably improving over Sarsa. Except for the small additional computational cost, Expected Sarsa may completely dominate both of the other more-well-known TD control algorithms.

### Maximization Bias and Double Learning
All the control algorithms that we have discussed so far involve maximization in the construction of their target policies. For example, in Q-learning the target policy is the greedy policy given the current action values, which is defined with a max, and in Sarsa the policy is often $\epsilon-greedy$, which also involves a maximization operation. In these algorithms, a maximum over estimated values is used implicitly as an estimate of the maximum value, which can lead to a significant positive bias. To see why, consider a single state s where there are many actions a whose true values, $q(s, a)$, are all zero but whose estimated values, $Q(s, a)$, are uncertain and thus distributed some above and some below zero. The maximum of the true values is zero, but the maximum of the estimates is positive, a positive bias. We call this maximization bias.

Are there algorithms that avoid maximization bias? To start, consider a bandit case in which we have noisy estimates of the value of each of many actions, obtained as sample averages of the rewards received on all the plays with each action. As we discussed above, there will be a positive maximization bias if we use the maximum of the estimates as an estimate of the maximum of the true values. One way to view the problem is that it is due to using the same samples (plays) both to determine the maximizing action and to estimate its value. Suppose we divided the plays in two sets and used them to learn two independent estimates, call them $Q_1(a)$ and $Q_2(a)$, each an estimate of the true value q(a), for all a 2 A. We could then use one estimate, say $Q_1(a)$, to determine the maximizing action $A\star = \max_a(Q_1(a))$, and the other, $Q_2(a)$, to provide the estimate of its value, $Q_2(A^\star) = Q_2(\max_a(Q1(a)))$. This estimate will then be unbiased in the sense that $E[Q2(A^\star)] = q(A^\star)$. We can also repeat the process with the role of the two estimates reversed to yield a second unbiased estimate $Q_1(\max_a(Q2(a)))$. This is the idea of double learning. Note that although we learn two estimates, only one estimate is updated on each play; double learning doubles the memory requirements, but does not increase the amount of computation per step.

The idea of double learning extends naturally to algorithms for full MDPs. For example, the double learning algorithm analogous to Q-learning, called Double Q-learning, divides the time steps in two, perhaps by flipping a coin on each step. If the coin comes up heads, the update is

 $$
Q_1(s_t, a_t) += \alpha(r_{t+1} + \gamma~ ~Q_1(s_{t+1}, \max_a(Q_2(s_{t+1})), a) - Q(s_t, a_t)) 
$$

**Double Q-Learning Algorithm:**
```
Inputs: an epsilon-greedy policy, pi that uses Q

Params: alpha, gamma

Initialization: q1, q2 for all states and q(terminal) = 0

while True:
	choose an s
	while s is not terminal:
		a = pi(s, q1 + q2) // the second input is not neccesserily this one
		r, s_next = env(a)
		if rand(0~1) > 0.5:
			q1(s, a) += alpha (r + gamma * q1(s_next, greedy(s_next, q2)) - q1(s, a))
		else:
			q2(s, a) += alpha (r + gamma * q2(s_next, greedy(s_next, q1)) - q2(s, a))
			s = s_next
```


### Open challenges
- If both TD and Monte Carlo methods converge asymptotically to the correct predictions, then a natural next question is “Which gets there first?” In other words, which method learns faster? Which makes the more efficient use of limited data? At the current time this is an open question in the sense that no one has been able to prove mathematically that one method converges faster than the other. In fact, it is not even clear what is the most appropriate formal way to phrase this question! In practice, however, TD methods have usually been found to converge faster than constant-$\alpha$ MC methods on stochastic tasks.


## $n$-step Bootstrapping
TODO

## Planning and Learning with Tabular Methods



## References
1. (Book) Reinforcement Learning: An Introduction [Sutton & Burto]
2. (Course) [Reinforcement learning specialization](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.coursera.org/specializations/reinforcement-learning&ved=2ahUKEwi-2-Prop2MAxUIcKQEHeEwDgcQFnoECBoQAQ&usg=AOvVaw1VX-UHhG8EU2QL8dIYAas4) 
