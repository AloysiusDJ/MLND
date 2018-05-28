import numpy as np
import random
from collections import namedtuple, deque
from keras import layers, models, optimizers
from keras import backend as K

# Create DDPG Agent
class DDPGAgent:
    def __init__(self, task, param):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Parameters
        self.buffer_size = param.get('buffer_size')
        self.batch_size = param.get('batch_size')
        self.gamma = param.get('gamma')
        self.tau = param.get('tau')
        self.exploration_mu = param.get('exploration_mu')
        self.exploration_theta = param.get('exploration_theta')
        self.exploration_sigma = param.get('exploration_sigma')
        
        # ActorPolicyNetwork
        self.actor_local = ActorPolicyNetwork(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = ActorPolicyNetwork(self.state_size, self.action_size, self.action_low, self.action_high)

        # CriticValueNetwork
        self.critic_local = CriticValueNetwork(self.state_size, self.action_size)
        self.critic_target = CriticValueNetwork(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # OUNoise
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # ReplayBuffer
        self.memory = ReplayBuffer(self.buffer_size)



    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
        self.last_state = next_state

    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())

    def learn(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        actions_next = self.actor_target.model.predict_on_batch(next_states)
        targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        targets = rewards + self.gamma * targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=targets)

        # Train LocalActor
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        # Update target
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        
    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        


# DDPG Actor Policy Network -> Maps states to Actions
class ActorPolicyNetwork:   
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.build_model()
    
    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        
        # Hidden layers
        h1 = layers.Dense(units=32, activation='relu',kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        h1 = layers.BatchNormalization()(h1)
        h2 = layers.Dense(units=64, activation='relu',kernel_regularizer=layers.regularizers.l2(1e-6))(h1)
        h2 = layers.BatchNormalization()(h2)
        #h3 = layers.Dense(units=128, activation='relu',kernel_regularizer=layers.regularizers.l2(1e-6))(h2)
        #h3 = layers.BatchNormalization()(h3)
        
        # Output Layer
        out = layers.Dense(units=self.action_size, activation='sigmoid', name='out')(h2)
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(out)
        
        # Create Model
        self.model = models.Model(inputs=states, outputs=actions)
        
        # Define loss function using Q values
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)
        
        # Define Optimizer and Training Function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], outputs=[], updates=updates_op)

        
# DDPG CriticValue Network -> Maps (state,action) pairs to Q-values
class CriticValueNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size 
        self.action_size = action_size
        self.build_model()
        
    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
            
        # Add hidden layers for States
        h1_states = layers.Dense(units=32, activation='relu',kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        h1_states = layers.BatchNormalization()(h1_states)
        h2_states = layers.Dense(units=64, activation='relu',kernel_regularizer=layers.regularizers.l2(1e-6))(h1_states)
        h2_states = layers.BatchNormalization()(h2_states)
        
        # Add hidden layers for Actions
        h1_actions = layers.Dense(units=32, activation='relu',kernel_regularizer=layers.regularizers.l2(1e-6))(actions)
        h1_actions = layers.BatchNormalization()(h1_actions)
        h2_actions = layers.Dense(units=64, activation='relu',kernel_regularizer=layers.regularizers.l2(1e-6))(h1_actions)
        h2_actions = layers.BatchNormalization()(h2_actions)
        
        # Combine State and Action Layers
        out = layers.Add()([h2_states, h2_actions])
        out = layers.Activation('relu')(out)
        out = layers.Dense(units=1, name='out')(out)
            
        # Create Model
        self.model = models.Model(inputs=[states, actions], outputs=out)
          
        # Define Optimizer and compile model
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
            
        # Compute action gradients
        action_gradients = K.gradients(out, actions)
            
        # Fetch action gradients
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)

        
# ReplayBuffer
class ReplayBuffer:  
    def __init__(self, size):
        self.size = size
        self.memory = deque(maxlen=self.size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])        

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

    
# OUNoise (Ornstein-Uhlenbeck)
class OUNoise: 
    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
