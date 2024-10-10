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
% eig(A - B*K)
