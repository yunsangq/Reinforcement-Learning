import numpy as np
import tensorflow as tf
import gym
from gym import wrappers

env = gym.make("Pong-v0")
env = wrappers.Monitor(env, './pong', force=True)

n_actions = env.action_space.n
gamma = 0.99
hidden = 200
n_features = 80 * 80
# Actor
tf_observations = tf.placeholder(tf.float32, [None, n_features], name="observation")
tf_actions = tf.placeholder(tf.int32, [None, ], name="action")
tf_delta = tf.placeholder(tf.float32, [None, ], name="delta")

W1 = tf.get_variable("W1", shape=[n_features, hidden],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([hidden]), name="b1")
layer1 = tf.nn.relu(tf.matmul(tf_observations, W1) + b1)
W1_1 = tf.get_variable("W1_1", shape=[hidden, n_actions],
                       initializer=tf.contrib.layers.xavier_initializer())
b1_1 = tf.Variable(tf.zeros([n_actions]), name="b1_1")

action_prob = tf.nn.softmax(tf.matmul(layer1, W1_1) + b1_1)

log_prob = tf.reduce_sum(-tf.log(action_prob)*tf.one_hot(tf_actions, n_actions), axis=1)
loss = tf.reduce_mean(log_prob*tf_delta)
train_op = tf.train.AdamOptimizer().minimize(loss)

# Critic
tf_observations_c = tf.placeholder(tf.float32, [None, n_features], name="observation_c")
tf_G_t = tf.placeholder(tf.float32, [None, 1], name="G_t")

W2 = tf.get_variable("W2", shape=[n_features, hidden],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros([hidden]), name="b2")
layer1_c = tf.nn.relu(tf.matmul(tf_observations_c, W2) + b2)
W2_1 = tf.get_variable("W2_1", shape=[hidden, 1],
                       initializer=tf.contrib.layers.xavier_initializer())
b2_1 = tf.Variable(tf.zeros([1]), name="b2_1")

V = tf.matmul(layer1_c, W2_1) + b2_1
delta = tf_G_t - V
loss_c = tf.reduce_sum(tf.square(tf_G_t - V))
train_op_c = tf.train.AdamOptimizer().minimize(loss_c)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def choose_action(observation):
    prob = sess.run(action_prob, feed_dict={tf_observations: np.reshape(observation, (1, n_features))})
    return np.random.choice(range(n_actions), p=prob[0])


def discount_norm_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    tmp = 0
    for t in reversed(range(0, len(rewards))):
        tmp = tmp * gamma + rewards[t]
        discounted_rewards[t] = tmp

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    return discounted_rewards


avg_reward = 0.
for i_episode in range(10010):
    observation = env.reset()

    observations = []
    rewards = []
    actions = []

    while True:
        '''
        if i_episode % 500 == 0:
            env.render()
        '''
        observation = prepro(observation)
        action = choose_action(observation)
        new_observation, reward, done, info = env.step(action)

        observations.append(observation)
        rewards.append(reward)
        actions.append(action)

        if done:
            discounted_reward = discount_norm_rewards(rewards)

            _, _delta = sess.run([train_op_c, delta], feed_dict={tf_observations_c: np.vstack(observations),
                                                                 tf_G_t: np.array([discounted_reward]).T})

            sess.run([train_op], feed_dict={tf_observations: np.vstack(observations),
                                            tf_actions: np.array(actions),
                                            tf_delta: _delta.flatten()})
            avg_reward += sum(rewards)
            print("Episode: {} Reward: {}".format(i_episode, sum(rewards)))
            if i_episode % 100 == 0 and i_episode != 0:
                print("Episode: {} Avg Reward: {}".format(i_episode, avg_reward / 100.))
                avg_reward = 0.
            break
        observation = new_observation

env.close()