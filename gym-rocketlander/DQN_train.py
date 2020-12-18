import numpy as np
import gym
import gym_rocketlander

import collections
import random
import cv2

import tensorflow as tf
from keras.models import Model
from keras import layers
from keras import optimizers
import pickle

Transition = collections.namedtuple('Transition', ['current_state', 'action', 'reward', 'next_state', 'done'])


class RocketAgent:
    def __init__(self, env, batch_size=32):
        self.env = env
        self.num_actions = 7  # full descriptions given in rocket_lander.py
        self.epsilon_range = (0.99, 0.05)  # serves as the probability divider between random action and greedy action
        self.epsilon = self.epsilon_range[0]  # this value will decay
        self.alpha = 0.1  # learning rate
        self.gamma = 0.99  # weight of new rewards
        self.memory = None  # Memory to store replay experience for DQN. Max = 200000
        self.original_img_height = 720
        self.original_img_width = 500
        self.img_height = int(0.2 * self.original_img_height)  # input image height (original scaled down by 0.2)
        self.img_width = int(0.2 * self.original_img_width)   # Input image width (original scaled down by 0.2)
        self.num_frames = 4  # Number of image samples to stack
        self.model = self.get_model()  # Initialize the primary neural network
        self.target_model = tf.keras.models.clone_model(self.model)  # create target network by copying primary network
        self.update_target_network()
        self.batch_size = batch_size

        self.action_mask = np.ones((1, self.num_actions))

    def decay_epsilon(self, current_epoch, total_epochs):
        rate = max(1e-5, (total_epochs - current_epoch)/total_epochs)
        self.epsilon = (self.epsilon_range[0] - self.epsilon_range[1]) * rate + self.epsilon_range[1]

    def get_model(self) -> Model:
        """
        Build CNN neural network for reading in Rocket lander image data
        :return: keras.models.Model compiled with optimizer and layers
        """
        input_shape = (self.num_frames, self.img_height, self.img_width)  # define input shape
        actions_input = layers.Input((self.num_actions,), name='action_mask')  # actions input
        frames_input = layers.Input(input_shape, name='input_layer')  # image input

        #image data layers
        conv_1 = layers.Conv2D(32, 8, strides=4, padding='same', activation='relu', name='Conv1')(frames_input)
        conv_2 = layers.Conv2D(64, 4, strides=2, padding='same', activation='relu', name='Conv2')(conv_1)
        conv_3 = layers.Conv2D(64, 3, strides=1, padding='same', name='Conv3', activation='relu')(conv_2)
        flatten_1 = layers.Flatten()(conv_3)
        dense_1 = layers.Dense(512, activation='relu', name='Hidden1')(flatten_1)
        output = layers.Dense(self.num_actions, activation='linear', name='Output')(dense_1)

        # image and actions data layers
        masked_output = layers.Multiply(name='Action_Mask')([output, actions_input])

        # compile the model and optimizer
        model = Model(inputs=[frames_input, actions_input], outputs=[masked_output])
        optimizer = optimizers.Adam(lr=self.alpha)
        model.compile(optimizer, loss=tf.losses.Huber())
        return model

    def process(self, image):
        # if an image is 3D, that indicates that it is RGB. If so, change to gray
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # rescale image
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        return image

    def remember(self, current_state, action, reward, next_state, done):
        # add a sample to the experience replay memory buffer
        self.memory.append(Transition(current_state, action, reward, next_state, done)) # for readability, encode each Transition

    def update_target_network(self):
        # copy the model weights into the target model
        self.target_model.set_weights(self.model.get_weights())

    def get_best_action(self, current_state) -> int:
        """
        Based off the current state, returns the action_ind with the highest Q-value.
        :param: current_state (1, 4, 100, 150)
        :return: a tuple encoding the action with the highest Q-value at the current state
        """
        current_state = current_state.astype('float32') / 255
        possible_actions = self.model.predict([current_state, self.action_mask])[0]
        best_action = np.argmax(possible_actions)
        return best_action

    def experience_replay(self):
        #extract all the required data from each batch
        samples = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*samples))
        current_states, actions, rewards, next_states, dones = map(np.stack, [*batch])
        current_states = current_states.astype('float32') / 255
        next_states = next_states.astype('float32') / 255
        batch_action_mask = np.tile(self.action_mask, (self.batch_size, 1))

        #Predict the q-values of all the actions in the next state using the target model
        q_vals = self.target_model.predict([next_states, batch_action_mask])

        #If the next state is the terminal state the q-value is zero. This is how the mountain car
        #problem hs been defined
        q_vals[dones] = 0

        #Calulate the Bellman target using the Bellman equation
        targets = rewards + self.gamma * np.max(q_vals, axis=1)

        #This action mask encodes your target value in a one-hot fashion so as to easily subtract
        #the two values while calculating the Bellman error.
        one_hot_actions = np.eye(self.num_actions)[actions]

        labels = one_hot_actions * targets[:, None]

        # Model.train_on_batch will first make a prediction using the current state and encode the output
        # in a one-hot fashion but with the q-value of the action as the hot entry instead of a '1'.
        # It will then calculate the loss using the predition and the given label and update the Network's
        # Weights accordingly.
        loss = self.model.train_on_batch([current_states, one_hot_actions], labels)
        return loss

    def save(self, path):
        # save the model
        self.model.save(path)

    def play(self, epochs=100):
        memory_buffer = collections.deque(maxlen=self.num_frames)
        done = False
        train = False
        update_interval = int(epochs/100) # we want to update 100 times over the course of the training
        save_interval = int(epochs/10) # we want to save 10 times over the course of the training
        max_frames_per_epoch = 300
        episode_rewards = []
        max_total_steps = max_frames_per_epoch * epochs
        self.memory = collections.deque(maxlen=int(0.01 * max_total_steps))
        min_experience = int(self.memory.maxlen * 0.75)

        for epoch in range(epochs):
            memory_buffer.clear()
            self.env.reset()
            image = self.env.render(mode='rgb_array')
            frame = self.process(image)
            frame = frame.reshape(1, frame.shape[0], frame.shape[1])
            state = np.tile(frame, (self.num_frames, 1, 1))
            memory_buffer.extend(state)
            action = self.env.action_space.sample()
            episode_reward = 0

            for t in range(max_frames_per_epoch):
                current_step = (epoch * max_frames_per_epoch) + (t + 1)
                if not t % self.num_frames:
                    # if timestep is 0 or every four timesteps/frames, train
                    if np.random.uniform() < self.epsilon:
                        # pick a random action
                        action = self.env.action_space.sample()
                        if train:
                            # decay epsilon
                            self.decay_epsilon(current_step, max_total_steps)
                    else:
                        # have to reshape state to fit into NN
                        action = self.get_best_action(state.reshape(1, state.shape[0], state.shape[1],
                                                                    state.shape[2]))
                # need to collect next batch of frames
                _, reward, done, _ = self.env.step(action)
                next_frame = self.process(self.env.render(mode='rgb_array'))
                memory_buffer.append(next_frame)

                next_state = np.stack(memory_buffer)
                self.remember(state, action, reward, next_state, done)

                state = next_state

                if not train and len(self.memory) >= min_experience:
                    train = True
                    print("training has commenced")

                if train:
                    self.experience_replay()

                episode_reward += reward

                if done:
                    break

            episode_rewards.append(episode_reward)
            if not epoch % int(epochs/100):
                print("Epoch: {}/{}, epsilon: {}, episode reward: {}".format(epoch, epochs, self.epsilon, episode_reward))

            if train:
                if not epoch % update_interval:
                    print("weights updated at epoch {}".format(epoch))
                    self.update_target_network()
                if not epoch % save_interval:
                    print("model saved at epoch {}".format(epoch))
                    self.save('./train/DQN_CNN_model_{}.h5'.format(epoch))
                    pickle.dump(episode_reward, open('./train/rewards_{}.dump'.format(epoch), 'wb'))

        self.env.close()


if __name__ == '__main__':
    env = gym.make('rocketlander-v0')
    agent = RocketAgent(env, batch_size=32)
    agent.play(epochs=int(1e3))









