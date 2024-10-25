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
