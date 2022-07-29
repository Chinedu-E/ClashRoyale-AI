from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Dense, LeakyReLU, Concatenate, Dropout



class ParamNet:
    """ Parameter Network for PDQN
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
        x = Dropout(0.2)(x)
        
        x = Dense(256, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        parameters = Dense(8, activation="sigmoid")(x)
    
        #parameters = Lambda(lambda i: i* self.act_range)(parameters)
        return Model(inputs = [input_a, input_b], outputs =parameters)
    
    def train(self, obs, critic):
        """ training Actor's Weights
		"""
        with tf.GradientTape() as tape:
            actions = self.network(obs)
            actor_loss = -tf.reduce_sum(critic([*obs,actions]))
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
        return self.target_network.predict(new_obs, verbose=0)
    
    def save_network(self, path):
        self.network.save_weights(path + '_actor.h5')
        self.target_network.save_weights(path +'_actor_t.h5')
        
    def load_network(self, path):
        self.network.load_weights(path + '_actor.h5')
        self.target_network.load_weights(path + '_actor_t.h5')