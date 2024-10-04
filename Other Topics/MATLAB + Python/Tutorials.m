clc; clear; close all

% This tutorial is based on this link on MATLAB wesite
% https://uk.mathworks.com/products/matlab/matlab-and-python.html

%% check if there is a supported version on your system or not
% Link : https://uk.mathworks.com/help/matlab/matlab_external/install-supported-python-implementation.html#buialof-39
pyenv

% you can see python supported version in this link : 
% https://uk.mathworks.com/support/requirements/python-compatibility.html

%% A simple example using numpy
cos_of_pi = py.numpy.cos(pi);
np_cos = py.importlib.import_module("numpy").cos;
cos_of_pi_2 = np_cos(pi);

% Link : https://uk.mathworks.com/help/matlab/matlab_external/create-object-from-python-class.html

%% Run a .py file and getting a variable as an output
% Link : https://uk.mathworks.com/help/matlab/ref/pyrunfile.html

a = pyrunfile("AdderClass.py", "simple_adder_class").a;
% I tried to get the class, However it didn't work properly here
% simple_adder_class = pyrunfile("AdderClass.py", "simple_adder_class");
% simple_adder_class.reset_vars()
% simple_adder_class.add_to_a(12)
% simple_adder_class.add_to_b(-6)
% simple_adder_class.print_a()
% simple_adder_class.print_b()

% So I guess each python code or script can be run only once and I can't
% Have a python class fully functional here on MATLAB. So disapointing!

% In conclusion, I can say that only paradigm available using python is
% functional programming.