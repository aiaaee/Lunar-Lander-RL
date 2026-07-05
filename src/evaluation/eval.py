from Environment.env import env
from IPython.display import HTML , Video
from base64 import b64encode
import imageio

frames = []
state  , _ = env.reset()
img = env.render()

for eps in range(1):
  state , _ = env.reset()
  done = False
  while not done :
    action = env.action_space.sample()
    state , reward , terminated , truncated , info = env.step(action)
    frames.append(env.render())
    done = terminated or truncated
    next_state = state

imageio.mimsave("LunarLander.mp4" , frames , fps=30)
Video("LunarLander.mp4" , embed=True)
