import gym
from gym import wrappers
import numpy as np
import tensorflow as tf

np.random.seed(1)
env = gym.make('LunarLander-v2')
env.seed(1)
env = wrappers.Monitor(env, './lld', force=True)


n_actions = env.action_space.n
n_features = env.observation_space.shape[0]
alpha = 0.05
beta = 0.01
gamma = 0.99

# Actor
tf_observations = tf.placeholder(tf.float32, [None, n_features], name="observation")
tf_actions = tf.placeholder(tf.int32, [None, ], name="action")
tf_delta = tf.placeholder(tf.float32, [None, ], name="delta")

W1 = tf.get_variable("W1", shape=[n_features, n_actions],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([n_actions]), name="b1")
action_prob = tf.nn.softmax(tf.matmul(tf_observations, W1) + b1)

log_prob = tf.reduce_sum(-tf.log(action_prob)*tf.one_hot(tf_actions, n_actions), axis=1)
loss = tf.reduce_mean(log_prob*tf_delta)
train_op = tf.train.AdamOptimizer(alpha).minimize(loss)

# Critic
tf_observations_c = tf.placeholder(tf.float32, [None, n_features], name="observation_c")
tf_G_t = tf.placeholder(tf.float32, [None, 1], name="G_t")

W2 = tf.get_variable("W2", shape=[n_features, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros([1]), name="b2")
V = tf.matmul(tf_observations_c, W2) + b2
delta = tf_G_t - V
loss_c = tf.reduce_sum(tf.square(tf_G_t - V))
train_op_c = tf.train.AdamOptimizer(beta).minimize(loss_c)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


def choose_action(observation):
    prob = sess.run(action_prob, feed_dict={tf_observations: np.reshape(observation, (1, n_features))})
    return np.random.choice(range(n_actions), p=prob[0])


def discount_norm_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    tmp = 0
    for t in reversed(range(0, len(rewards))):
        tmp = tmp*gamma+rewards[t]
        discounted_rewards[t] = tmp

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    return discounted_rewards

avg_reward = 0.
for i_episode in range(5001):
    observation = env.reset()

    observations = []
    rewards = []
    actions = []

    while True:
        if i_episode % 500 == 0:
            env.render()
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
            if i_episode % 100 == 0 and i_episode != 0:
                print("Episode: {} Reward: {}".format(i_episode, avg_reward / 100.))
                avg_reward = 0.
            break
        observation = new_observation

env.close()
