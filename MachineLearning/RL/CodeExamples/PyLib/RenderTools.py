from IPython.display import Video, display
import gymnasium as gym
import cv2

def render_gym_env(agent, env_name, num_timesteps = 400):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    video_path = f"Renders/Render_{agent.name}.mp4"
    frame_width, frame_height = env.render().shape[1], env.render().shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (frame_width, frame_height))
    for _ in range(num_timesteps):
        action = agent.get_action(state)
        state, _, _, _, _ = env.step(action)
        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        out.write(frame_bgr)
    out.release()
    env.close()

def sim_gym_env(agent, env_name, num_timesteps = 400):
    env = gym.make(env_name, render_mode='human')
    state, _ = env.reset()
    for _ in range(num_timesteps):
        action = agent.get_action(state)
        state, _, ter, tr, _ = env.step(action)
        if ter or tr:
            break
        env.render()
    env.close()