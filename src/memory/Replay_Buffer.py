from collections import deque 

class ReplayBuffer:
  def __init__(self , capacity):
    self.buffer = deque(maxlen=capacity)
  def __len__(self):
    return len(self.buffer)
  def add(self , state , action , reward , next_state , done):
    return self.buffer.append((state , action , reward , next_state , done))
  def sample(self , batch_size):
    batch = random.sample(self.buffer , batch_size)
    states , actions , rewards , next_states , dones = zip(*batch)
    return (
        np.array(states) ,
        np.array(actions) ,
        np.array(rewards) ,
        np.array(next_states) ,
        np.array(dones)
    )

