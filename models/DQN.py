from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from keras.layers import Input,Dense, Activation, BatchNormalization, Concatenate, Lambda
from keras.initializers import GlorotNormal
from keras.regularizers import l2
from keras.losses import MSE


class DDQNet:
    """ Double Duelling Q Network for PDQN
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
        """ Create a Deep Q Network with duelling layers using Keras
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
        h1_ = Dense(512, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(concat)
        h1_b = BatchNormalization()(h1_)
        h1 = Activation('relu')(h1_b)
        
        # hidden_layer 2
        h2_ = Dense(256, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(h1)
        h2_b = BatchNormalization()(h2_)
        h2 = Activation('relu')(h2_b)
        
        # hidden_layer 3
        h3_ = Dense(128, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(h2)
        h3_b = BatchNormalization()(h3_)
        h3 = Activation('relu')(h3_b)
        
        #value stream
        value_fc = Dense(32, activation='relu', name="value_fc")(h3)
        value = Dense(1, activation='linear', name="value")(value_fc)
        
        #advantage stream
        advantage_fc = Dense(32, activation='relu', name='advantage_fc')(h3)
        advantage = Dense(5, activation='linear')(advantage_fc)
        
        def d_output(x):
            a = x[0]
            v = x[1]
            return v + tf.math.subtract(a, tf.math.reduce_mean(a, axis=1, keepdims=True))
        
        
        output = Lambda(d_output)([advantage, value])
        
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
        return self.network.predict(obs, verbose=0)
    
    def target_predict(self, new_obs):
        """Predict target Q-value from approximation function(Q-network)
		"""
        return self.target_network.predict(new_obs, verbose=0)
    
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