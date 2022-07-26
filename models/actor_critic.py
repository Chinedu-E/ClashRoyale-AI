from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from keras.layers import Input,Dense, Activation, BatchNormalization, LeakyReLU, Concatenate
from keras.initializers import GlorotNormal
from keras.regularizers import l2
from keras.losses import MSE
  

  
# now we can import the module in the parent
# directory.

class ActorNet:
    """ Actor Network for PDQN
	"""
    def __init__(self ,in_dim, out_dim, act_range, lr_, tau_):
        self.obs_dims = in_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.lr=lr_; self.tau = tau_
        
        #initializing actor network and target network
        self.network = self.create_network()
        self.target_network = self.create_network()
        self.optimizer = Adam(self.lr)
        
        #copy the weights for initialization
        weights_ = self.network.get_weights()
        self.target_network.set_weights(weights_)
        
    def create_network(self):
        """ Create a Actor Network Model using Keras
		"""
        #input layer (image observations in latent space from VAE)
        input_a = Input(shape=self.obs_dims[0])
        #input layer (card features)
        input_b = Input(shape=self.obs_dims[1])
        #combine inputs
        combined = Concatenate()([input_a, input_b])
        
        
        #branch 2
        fc = Dense(512, kernel_initializer='he_uniform')(combined)
        x = LeakyReLU()(fc)
        x = Dense(256, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        x = Dense(128, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        parameters = Dense(8, activation="sigmoid")(x)
    
        #parameters = Lambda(lambda i: i* self.act_range)(parameters)
        return Model(inputs = [input_a, input_b], outputs =parameters)
    
    def train(self, obs, critic):
        """ training Actor's Weights
		"""
        with tf.GradientTape() as tape:
            actions = self.network(obs)
            actor_loss = -tf.reduce_mean(critic([*obs,actions]))
        actor_grad = tape.gradient(actor_loss,self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grad,self.network.trainable_variables))
        
    def target_update(self):
        """ soft target update for training target actor network
		"""
        weights, weights_t = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(weights)):
            weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
        self.target_network.set_weights(weights_t)
        
    def predict(self, obs):
        """ predict function for Actor Network
		"""
        return self.network(obs)
    
    def target_predict(self, new_obs):
        """  predict function for Target Actor Network
		"""
        return self.target_network.predict(new_obs)
    
    def save_network(self, path):
        self.network.save_weights(path + '_actor.h5')
        self.target_network.save_weights(path +'_actor_t.h5')
        
    def load_network(self, path):
        self.network.load_weights(path + '_actor.h5')
        self.target_network.load_weights(path + '_actor_t.h5')
        
        
        
class CriticNet:
    """ Critic Network for PDQN
	"""
    def __init__(self, in_dim, out_dim, lr_, tau_, discount_factor):
        self.obs_dims = in_dim
        self.act_dim = out_dim
        self.lr = lr_; self.discount_factor=discount_factor;self.tau = tau_
        
        # initialize critic network and target network 
        self.network = self.create_network()
        self.target_network = self.create_network()
        
        self.optimizer = Adam(self.lr)
        
        # copy the weights for initialization
        weights_ = self.network.get_weights()
        self.target_network.set_weights(weights_)
        self.critic_loss = None
        self.loss = MSE
        
    def create_network(self):
        """ Create a Critic Network Model using Keras
			as a Q-value approximator function
		"""
        #input layer (image observations in latent space from VAE)
        input_a = Input(shape=self.obs_dims[0])
        #input layer (card features)
        input_b = Input(shape=self.obs_dims[1])
        #input layer (parameters)
        input_c = Input(shape=(8,))
        
        
        inputs = [input_a, input_b, input_c]
        concat = Concatenate(axis=-1)(inputs)
        
        # hidden layer 1
        h1_ = Dense(300, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(concat)
        h1_b = BatchNormalization()(h1_)
        h1 = Activation('relu')(h1_b)
        
        # hidden_layer 2
        h2_ = Dense(400, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(h1)
        h2_b = BatchNormalization()(h2_)
        h2 = Activation('relu')(h2_b)
        
        # output layer(actions)
        output_ = Dense(5, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(h2)
        output_b = BatchNormalization()(output_)
        output = Activation('linear')(output_b)
        
        return Model(inputs,output)
    
    def Qgradient(self, obs, acts):
        acts = tf.convert_to_tensor(acts)
        with tf.GradientTape() as tape:
            tape.watch(acts)
            q_values = self.network([*obs,acts])
            q_values = tf.squeeze(q_values)
        return tape.gradient(q_values, acts)
    
    def train(self, obs, acts, target):
        """Train Q-network for critic on sampled batch
		"""
        with tf.GradientTape() as tape:
            q_values = self.network([*obs, acts], training=True)
            critic_loss = self.loss(q_values, target)
            self.critic_loss = critic_loss
        critic_grad = tape.gradient(critic_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_grad, self.network.trainable_variables))
        
    def predict(self, obs):
        """Predict Q-value from approximation function(Q-network)
		"""
        return self.network.predict(obs)
    
    def target_predict(self, new_obs):
        """Predict target Q-value from approximation function(Q-network)
		"""
        return self.target_network.predict(new_obs)
    
    def target_update(self):
        """ soft target update for training target critic network
		"""
        weights, weights_t = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(weights)):
            weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
        self.target_network.set_weights(weights_t)
        
    def save_network(self, path):
        self.network.save_weights(path + '_critic.h5')
        self.target_network.save_weights(path + '_critic_t.h5')
        
    def load_network(self, path):
        self.network.load_weights(path + '_critic.h5')
        self.target_network.load_weights(path + '_critic_t.h5')
        
        
