import numpy as np
from game import BouncyBalls
from model import *

class Env(object):
    r"""The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return

    @property
    def unwrapped(self):
        """Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        self.close()
        # propagate exception
        return False
    
    
class dummy_env(Env):
    def __init__(self):
        super().__init__()
        self.action_space = np.zeros(3)
        self.observation_space = np.zeros(1)
        print('created')
        
    def step(self, action):
        obs = np.zeros(1)
        reward = action[0]
        done = True
        info = {}
        #print(reward)
        return obs, reward, done, info
    
    def reset(self):
        obs = np.zeros(1)
        return obs
    
    
class ball_env_1(Env):
    """
    Ball environment #1
    Only one time step.
    """
    def __init__(self):
        super().__init__()
        self.action_space = np.zeros(6)
        self.observation_space = np.zeros(1)
        self.game = BouncyBalls()
        print('created')
        
    def step(self, action):
        obs = np.zeros(1)
        ball_posi = self.game.run_one_episode(action)
        reward = self.posi_reward(ball_posi)
        done = True
        info = {}
        #print(reward)
        return obs, reward, done, info
    
    def posi_reward(self, posi):
        reward = 0.0
        if posi[0] >= 550 and posi[1]> 150:
            reward = 1.0
        return reward
        
    
    def reset(self):
        obs = np.zeros(1)
        return obs
    
    
class ball_env_2(Env):
    """
    Ball environment #2
    Three time steps, each for one platform.
    """
    action_space = np.zeros(2)
    observation_space = np.zeros(9)
    def __init__(self):
        super().__init__()
        #self.action_space = np.array([0,0])
        #self.observation_space = np.array([0,0,0,0,0,0,0,0,0])
        self.game = BouncyBalls()
        self.step_count = 0
        self.cumulative_action = np.zeros(6)
        print('created')
        
    def step(self, action):
        self.step_count += 1
        
        if self.step_count >= 3:
            # let the ball go
            ball_posi = self.game.run_one_episode(self.cumulative_action)
            reward = self.posi_reward(ball_posi)
            done = True
            # reset
            self.step_count = 0
            self.cumulative_action = np.zeros(6)
        else:
            self.cumulative_action[self.step_count*2-2:self.step_count*2] = action
            reward = 0
            done = False
            
        obs = np.zeros(9)
        for i in range(self.step_count):
            obs[i*3] = 1
            obs[i*3+1:i*3+3] = action
        info = {}
        #print(reward)
        return obs, reward, done, info
    
    def posi_reward(self, posi):
        reward = 0.0
        if posi[0] >= 550 and posi[1]> 150:
            reward = 1.0
        return reward
        
    
    def reset(self):
        obs = self.observation_space
        return obs
        
class ball_env_3(Env):
    """
    Ball environment #3
    Three time steps, each for one platform. With predicted ball position as observation.
    """
    action_space = np.zeros(2)
    observation_space = np.zeros(11)
    def __init__(self):
        super().__init__()
        #self.action_space = np.array([0,0])
        #self.observation_space = np.array([0,0,0,0,0,0,0,0,0])
        self.game = BouncyBalls()
        self.step_count = 0
        self.cumulative_action = np.zeros(6)
        print('created')
        
    def step(self, action):
        self.step_count += 1
        
        if self.step_count >= 3:
            # let the ball go
            ball_posi = self.game.run_one_episode(self.cumulative_action)
            reward = self.posi_reward(ball_posi)
            done = True
            # reset
            self.step_count = 0
            self.cumulative_action = np.array([0,0,0,0,0,0])
        else:
            self.cumulative_action[self.step_count*2-2:self.step_count*2] = action
            reward = 0
            done = False
            
        obs = np.array([0,0,0,0,0,0,0,0,0])
        for i in range(self.step_count):
            obs[i*3] = 1
            obs[i*3+1:i*3+3] = action
        info = {}
        #print(reward)
        return obs, reward, done, info
    
    def posi_reward(self, posi):
        reward = 0.0
        if posi[0] >= 550 and posi[1]> 150:
            reward = 1.0
        return reward
        
    
    def reset(self):
        obs = self.observation_space
        return obs
        
class ball_env_4(Env):
    """
    Ball environment #4
    Ten time steps. At each time step, change position or let the ball go. No prediction model.
    """
    action_space = np.zeros(6+1)# posi for three platforms + let the ball go
    observation_space = np.zeros(7)
    def __init__(self):
        super().__init__()
        #self.action_space = np.array([0,0])
        #self.observation_space = np.array([0,0,0,0,0,0,0,0,0])
        self.game = BouncyBalls()
        self.step_count = 0
        self.cumulative_action = np.zeros(6)
        print('created')
        
    def step(self, action):
        self.step_count += 1
        
        if action[6] > 0 or self.step_count >= 10:
            # let the ball go
            obs = action
            ball_posi = self.game.run_one_episode(self.cumulative_action)
            
            reward = self.posi_reward(ball_posi)
            # special case for leting the ball go on first step
            if self.step_count == 1:
                reward = 0
            done = True
            
            # reset
            self.step_count = 0
            #self.cumulative_action = np.array([0,0,0,0,0,0])
        else:
            self.cumulative_action = action
            obs = action
            reward = 0
            done = False
            
        #obs = np.array([0,0,0,0,0,0,0,0,0])
        #for i in range(self.step_count):
        #    obs[i*3] = 1
        #    obs[i*3+1:i*3+3] = action
        info = {}
        #print(reward)
        return obs, reward, done, info
    
    def posi_reward(self, posi):
        reward = 0.0
        if posi[0] >= 550 and posi[1]> 150:
            reward = 1.0
        return reward
        
    
    def reset(self):
        obs = self.observation_space
        return obs
        
        
class ball_env_5(Env):
    """
    Ball environment #5
    Ten time steps. At each time step, change position or let the ball go. With predicted ball position as observation.
    """
    action_space = np.zeros(6+1)# posi for three platforms + let the ball go
    observation_space = np.zeros(7+2)# action + predicted ball posi
    def __init__(self):
        super().__init__()
        #self.action_space = np.array([0,0])
        #self.observation_space = np.array([0,0,0,0,0,0,0,0,0])
        self.game = BouncyBalls()
        self.step_count = 0
        #self.cumulative_action = np.zeros(6)
        self.pred_net = LSTM_Init_To_Many_3()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pred_net.load_state_dict(torch.load('preTrained/CP_epoch30.pth', map_location=self.device))
        self.pred_net.to(device=self.device)
        self.pred_net.eval()
        print('created')
        
    def step(self, action):
        self.step_count += 1
        
        if action[6] > 0 or self.step_count >= 10:
            # let the ball go
            ball_posi = self.game.run_one_episode(action)
            reward = self.posi_reward(ball_posi)
            # special case for leting the ball go on first step
            if self.step_count == 1:
                reward = 0
            done = True
            # reset
            self.step_count = 0
            #self.cumulative_action = np.array([0,0,0,0,0,0])
        else:
            #self.cumulative_action[self.step_count*2-2:self.step_count*2] = action
            reward = 0
            done = False
            
        # prediction 
        mean = np.array([ 30.893, 270.33,  200.388, 199.573, 350.057, 200.53 ])
        std = np.array([14.54288661, 14.70269023, 14.31668453, 14.40488358, 14.85717843, 15.25080654])
        normalized_platform_posi = (action[:6]-mean)/std
        with torch.no_grad():
            pred_input = torch.from_numpy(np.expand_dims(normalized_platform_posi, axis=0)).float().to(self.device)
            pred_output = self.pred_net(pred_input).cpu().numpy()
            
        def get_pred_ball_posi(pred_output, x_min=20, x_max=550,  y_min=50, y_max=550):
            mean = np.array([163.29437530326638, 279.7768839198992])
            std = np.array([138.14349185245848, 113.09608505385799])
            last_posi =pred_output[0,-1]
            pred_output_denormlilzed = pred_output*std+mean
            for i in range(pred_output.shape[1]):
                if pred_output_denormlilzed[0,i,0] < x_min or pred_output_denormlilzed[0,i,0] > x_max or pred_output_denormlilzed[0,i,1] < y_min or pred_output_denormlilzed[0,i,1] > y_max:
                    last_posi =pred_output[0,i]
                    break
            return last_posi
        
        last_posi = get_pred_ball_posi(pred_output)

        obs = np.zeros(9)
        obs[:7]=action
        obs[7:]=last_posi
        info = {}
        #print(reward)
        return obs, reward, done, info
    
    def posi_reward(self, posi):
        reward = 0.0
        if posi[0] >= 550 and posi[1]> 150:
            reward = 1.0
        return reward
        
    
    def reset(self):
        obs = self.observation_space
        return obs
    
        
        
class ball_env_6(Env):
    """
    Ball environment #6
    Ten time steps. At each time step, change position or let the ball go. With ORACLE predicted ball position as observation.
    """
    action_space = np.zeros(6+1)# posi for three platforms + let the ball go
    observation_space = np.zeros(7+2)# action + predicted ball posi
    def __init__(self):
        super().__init__()
        #self.action_space = np.array([0,0])
        #self.observation_space = np.array([0,0,0,0,0,0,0,0,0])
        self.game = BouncyBalls()
        self.step_count = 0
        #self.cumulative_action = np.zeros(6)
        self.pred_net = LSTM_Init_To_Many_3()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pred_net.load_state_dict(torch.load('preTrained/CP_epoch30.pth', map_location=self.device))
        self.pred_net.to(device=self.device)
        self.pred_net.eval()
        print('created')
        
    def step(self, action):
        self.step_count += 1
        
        if action[6] > 0 or self.step_count >= 10:
            # let the ball go
            ball_posi = self.game.run_one_episode(action)
            reward = self.posi_reward(ball_posi)
            # special case for leting the ball go on first step
            if self.step_count == 1:
                reward = 0
            done = True
            # reset
            self.step_count = 0
            #self.cumulative_action = np.array([0,0,0,0,0,0])
        else:
            #self.cumulative_action[self.step_count*2-2:self.step_count*2] = action
            reward = 0
            done = False
            
        ball_posi = self.game.run_one_episode(action)   
        last_posi = (ball_posi-np.array([300,300]))/np.array([300,300])

        obs = np.zeros(9)
        obs[:7]=action
        obs[7:]=last_posi
        info = {}
        #print(reward)
        return obs, reward, done, info
    
    def posi_reward(self, posi):
        reward = 0.0
        if posi[0] >= 550 and posi[1]> 150:
            reward = 1.0
        return reward
        
    
    def reset(self):
        obs = self.observation_space
        return obs