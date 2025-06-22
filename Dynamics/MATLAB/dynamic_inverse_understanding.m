clc;clear;close all

addpath("../MLib/packages/MATLAB/mr")

% At rest:
% o---------
%  i,0,    L = 1

% After rotating
%    /
%   /
%  /
% o
%  i,0


M = [1 0 0 1
     0 1 0 0
     0 0 1 0
     0 0 0 1];
A = [0;1;0;0;0;-1];

% After 90 deg of rotating
T = M * expm(VecTose3(A) * pi/2)