from models.actor_critic import ActorNet, CriticNet
from utils.buffer import MemoryBuffer
from utils.noise import OUActionNoise
from utils.config import CONV_FILTERS, CONV_KERNEL_SIZES, CONV_STRIDES, CONV_T_FILTERS, CONV_T_KERNEL_SIZES, CONV_T_STRIDES, Z_DIM, NORMALIZE, INPUT_DIM
from models.autoencoder import VariationalAutoencoder as VAE
import numpy as np
import tensorflow as tf
import random





class PDQN:
    """
    PDQN actor-critic agent for parameterised action spaces
    [Xiong et al., 2018]
    """
    
    def __init__(self, env_, batch_size=32, w_per=False, epsilon=0.95, eps_decay=0.05, eps_min = 0.3, seed=None):
        self.env_ = env_
        self.obs_dim = [(Z_DIM,), env_.observation_space['features'].shape]
        self.act_dim = env_.action_space['position'].shape
        self.action_bound = tf.convert_to_tensor((env_.action_space['position'].high - env_.action_space['position'].low) / 2, dtype=tf.float32)
        self.max_action = env_.action_space['position'].high
        self.min_action = env_.action_space['position'].low
        self.action_shift = (env_.action_space['position'].high + env_.action_space['position'].low) / 2
        
        #setting seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        
        #creating/loading VAE
        self.vae = VAE(INPUT_DIM, CONV_FILTERS, CONV_KERNEL_SIZES, CONV_STRIDES,
                       CONV_T_FILTERS, CONV_T_KERNEL_SIZES, CONV_T_STRIDES, Z_DIM, use_batch_norm=True)
        self.vae.load_weights(r"E:\RL\Clash Royale\workspace\models\weights\vae7.h5")
        
        
        # initialize actor & critic and its targets
        self.discount_factor = 0.99
        self.actor = ActorNet(self.obs_dim, self.act_dim, self.action_bound, lr_=1e-4,tau_=1e-3)
        self.critic = CriticNet(self.obs_dim, self.act_dim, lr_=1e-3,tau_=1e-3,discount_factor=self.discount_factor)
        
        # Experience Buffer
        self.with_per = w_per #Priority Experience Replay
        self.buffer = MemoryBuffer(10000, with_per=w_per)
        self.batch_size = batch_size
        # OU-Noise-Process
        self.noise = OUActionNoise(size=8)
        
    ###################################################
    # Network Related
    ###################################################
    def make_action(self, obs, t, noise=True):
        """ predict next action from Actor's Policy
		"""
        z_space = self.vae.encoder(obs[0].reshape(1, *INPUT_DIM))
        card_feat = obs[1].reshape(1, self.obs_dim[1][0])
        
        parameters_ = self.actor.predict([z_space, card_feat])

        disc = self.critic.predict([z_space, card_feat, parameters_])[0]
        
        parameters_= np.array(parameters_)
        param = np.clip(parameters_ + (self.noise.generate(t) if noise else 0), 0, 1)
        
        rnd = np.random.uniform()
        if rnd < self.epsilon:
            #Exploration
            disc = np.zeros(1) #We explore by saving elixir, i.e do nothing.
            return (disc, np.zeros(8))
        else:
            #Exploitation
            disc = np.argmax(disc) # Highest Q-value
            
        return (np.array([disc]), param.flatten())
    
    def update_networks(self, obs, acts, critic_target):
        """ Train actor & critic from sampled experience
        """ 
        z_dim = self.vae.encoder.predict(np.array(obs[0], dtype=np.float32))
        # update critic
        obs[1] = np.array(obs[1], dtype=np.float32)
        acts = np.array(acts, dtype=np.float32)
        obs = [z_dim, obs[1]]
        self.critic.train(obs, acts, critic_target)
        
        # get next action and Q-value Gradient
        n_actions = self.actor.network.predict(obs)
        q_grads = self.critic.Qgradient(obs, n_actions)
        
        # update actor
        self.actor.train(obs,self.critic.network)
        
        # update target networks
        self.actor.target_update()
        self.critic.target_update()
        
    def replay(self, replay_num_):
        if self.with_per and (self.buffer.size() <= self.batch_size): return
        if (self.buffer.size() <= self.batch_size): return
        
        for _ in range(replay_num_):
            # sample from buffer
            states1,states2, actions, rewards, dones, new_states1,new_states2, idx = self.sample_batch(self.batch_size)
            
            #print(new_states1.shape)
            z_dim = self.vae.encoder.predict(np.array(new_states1, dtype=np.float32))
            new_states2 = np.array(new_states2, dtype=np.float32)
            new_states = [z_dim, new_states2]
            
            # get target q-value using target network
            q_vals = self.critic.target_predict([z_dim, new_states2, self.actor.target_predict(new_states)])
            
            # bellman iteration for target critic value
            critic_target = np.asarray(q_vals)
            for i in range(q_vals.shape[0]):
                if dones[i]:
                    critic_target[i] = rewards[i]
                else:
                    critic_target[i] = rewards[i] + self.discount_factor * np.max(q_vals[i])
                    
                if self.with_per:
                    self.buffer.update(idx[i], abs(q_vals[i]-critic_target[i]))
                    
            # train(or update) the actor & critic and target networks
            self.update_networks([states1, states2], actions, critic_target)
            
            
    ####################################################
    # Buffer Related
    ####################################################
    
    def memorize(self,obs,act,reward,done,new_obs):
        """store experience in the buffer"""
        if self.with_per:
            z_space = self.vae.encoder(obs[0].reshape(1, *INPUT_DIM))
            card_feat = obs[1].reshape(1, self.obs_dim[1][0])
            q_val = self.critic.network([z_space, card_feat,self.actor.predict([z_space, card_feat])])
            
            new_z_space = self.vae.encoder(obs[0].reshape(1, *INPUT_DIM))
            new_card_feat = obs[1].reshape(1, self.obs_dim[1][0])
            next_action = self.actor.target_network.predict([new_z_space, new_card_feat])
            q_val_t = self.critic.target_predict([new_z_space, new_card_feat, next_action])
            new_val = reward + self.discount_factor * q_val_t
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0		
        self.buffer.memorize(obs[0],obs[1],act,reward,done,new_obs[0],new_obs[1],td_error)
        
    def on_episode_end(self):
        """Performing epsiolon greedy parameter decay
        """
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_decay
        else:
            self.epsilon = self.eps_min
        
    def sample_batch(self, batch_size):
        """ Sampling from the batch
        """
        return self.buffer.sample_batch(batch_size)
    
    def save_weights(self,path):
        self.actor.save_network(path)
        self.critic.save_network(path)
        
    def load_weights(self, pretrained):
        self.actor.load_network(pretrained)
        self.critic.load_network(pretrained)
        
        
