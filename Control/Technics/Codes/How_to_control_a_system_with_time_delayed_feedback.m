clc;clear;close all

% Contiuous Model & Feedback Control Gain =================================
A = 0.0;
B = 1;
delay_of_angle = 0.5;
delay_of_velocity = 0.001;
K = 100;

control_freq = 10;
% Simulation Settings =====================================================
simple_policy  = false;
use_estimation = false;
dt_func = @() 0.001;
control_interval = @() 1/control_freq + .75/control_freq * (rand - 0.5);
feedback_noise_model = @() 0.0 * randn;
feedback_time_delay_and_noise_model = @(dt) 0.1;
% -------------------------------------------------------------------------
t  = 0;
last_control_call_time = -inf;
x = 0;
y = [];
u = [];
ref = [];
delayed_x_feedback_buffer.data = [];
delayed_x_feedback_buffer.time = [];
delayed_x_dot_feedback_buffer = delayed_x_feedback_buffer;
time_array_s = [];

memory.control_time_array = [];
memory.control_out_array  = [];
memory.control_time_array = [];
memory.predicted_x = [];
memory.after_delay = false;
while t <= 10
    dt = dt_func();

    % Reference ===========================================================
    setpoint = 1 + 1 * (t > 2) - 2 * (t > 5) - 1 * (t > 8);

    % Fill up memories ====================================================
    x_pre = x;
    delayed_x_feedback_buffer.data = [delayed_x_feedback_buffer.data; x_pre];
    delayed_x_feedback_buffer.time = [delayed_x_feedback_buffer.time; t];
    y = [y; x];
    ref = [ref; setpoint];
    
    if t >= delay_of_angle + feedback_time_delay_and_noise_model(dt)
        feedback      = delayed_x_feedback_buffer.data(1);
        feedback_time = delayed_x_feedback_buffer.time(1);

        delayed_x_feedback_buffer.data = delayed_x_feedback_buffer.data(2:end);
        delayed_x_feedback_buffer.time = delayed_x_feedback_buffer.time(2:end);
    else
        feedback = 0;
        feedback_time = 0;
    end
    
    feedback = feedback + feedback_noise_model();
    feedback_time = feedback_time;

    % Estimation Step =====================================================
    
    % Control Policy ======================================================
    if((t - last_control_call_time) > control_interval())
        if simple_policy
            u = [u; simple_control_policy(K, setpoint - feedback)];
        else
            [control_signal, memory] = control_policy(t, K, setpoint, feedback, feedback_time, memory);
            u = [u; control_signal];
        end
        last_control_call_time = t;
    else
        u = [u; u(end)];
        if ~simple_policy
            memory.predicted_x = [memory.predicted_x; memory.predicted_x(end)];
        end
    end
    
    % Simulate One Step ===================================================
    % dt = dt_func(t);
    A_d = expm(A * dt);
    if A ~= 0
        B_d = ((expm(A * dt) - 1) / A) * B;
    else
        B_d = dt;
    end

    x = A_d * x_pre + B_d * u(end);

    time_array_s = [time_array_s; t];

    t = t + dt;
end

subplot(211)
plot(time_array_s, ref, "LineWidth", 2)
hold on
plot(time_array_s, y, "LineWidth", 2)
legend("Ref", "Output", "Location","best")

subplot(212)
plot(time_array_s, u, "LineWidth", 2)

% =========================================================================
% =========================================================================
% =========================================================================
% =========================================================================
% =========================================================================
% =========================================================================
% =========================================================================
% =========================================================================

function des_vel = simple_control_policy(K, err)
    des_vel = K * err;
end

function [des_velocity, memory] = control_policy(time_s, K, des_angle, delayed_angle, delayed_angle_sample_time, memory)
    % Nominal Contiuous Model & Feedback Control Gain
    A = 0;
    B = 1;
    
    memory.control_time_array = [memory.control_time_array; time_s];
    
    if memory.after_delay || memory.control_time_array(1) < (delayed_angle_sample_time - 1 / 25)
        % predictive control
        memory.control_time_array(memory.control_time_array <= delayed_angle_sample_time) = delayed_angle_sample_time;
    
        dt_array = diff(memory.control_time_array);
        x = delayed_angle;
        for i = 1 : length(dt_array)
            A_d = expm(A * dt_array(i));
            if A ~= 0
                B_d = ((expm(A * dt_array(i)) - 1) / A) * B;
            else
                B_d = dt_array(i);
            end
            
            x = A_d * x + B_d * memory.control_out_array(i);
        end
        K = min(1/dt_array(end), K);
        pole_des = exp(-K*mean(dt_array));
        K_d   = (A_d - pole_des) / B_d;
        des_velocity = K_d * (des_angle - x); 
    
        % time array clean up
        memory.control_time_array = memory.control_time_array(2:end);
        memory.control_out_array  = memory.control_out_array(2:end);
        memory.predicted_x = [memory.predicted_x; x];
        memory.after_delay = true;
    else
        % normal control policy
        des_velocity = 0.0 * K * (des_angle - delayed_angle);  
        memory.predicted_x = [memory.predicted_x; 0];
    end
    
    memory.control_out_array = [memory.control_out_array; des_velocity];
end