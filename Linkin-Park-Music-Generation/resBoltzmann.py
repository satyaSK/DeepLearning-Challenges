#This file is heavily based on Daniel Johnson's midi manipulation code in https://github.com/hexahedria/biaxial-rnn-music-composition
# Also credits to Dan shiebler 
import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
from midi_manipulation import *


mysongs = get_songs('linkin_park_midi') #These songs have already been converted from midi to msgpack
print ("{} songs processed".format(len(mysongs)))


lowest_note = midi_manipulation.lowerBound 
highest_note = midi_manipulation.upperBound 
note_range = highest_note-lowest_note 

num_timesteps  = 15 
n_visible      = 2*note_range*num_timesteps 
n_hidden       = 50

num_epochs = 300
batch_size = 200 
learning_rate = tf.constant(0.005, tf.float32) 

x  = tf.placeholder(tf.float32, [None, n_visible], name="x") #The placeholder variable that holds our data
W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W") #The weight matrix that stores the edge weights
bh = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="bh")) #The bias vector for the hidden layer
bv = tf.Variable(tf.zeros([1, n_visible],  tf.float32, name="bv")) #The bias vector for the visible layer

def sample_it(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def get_gibbs_sample(k):
    
    def gibbs_step(count, k, xk):
        
        hk = sample_it(tf.sigmoid(tf.matmul(xk, W) + bh)) #Propagate the visible values to sample the hidden values
        xk = sample_it(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)) #Propagate the hidden values to sample the visible values
        return count+1, k, xk

    #gibbs steps for k iterations
    ct = tf.constant(0) # my counter
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), x])
    x_sample = tf.stop_gradient(x_sample) 
    return x_sample


x_sample = get_gibbs_sample(1) 
h = sample_it(tf.sigmoid(tf.matmul(x, W) + bh)) 
h_sample = sample_it(tf.sigmoid(tf.matmul(x_sample, W) + bh)) 


size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder  = tf.multiply(learning_rate/size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
bh_adder = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]




with tf.Session() as sess:
    init = sess.run(tf.global_variables_initializer())
    
    for epoch in tqdm(range(num_epochs)):
        for song in mysongs:
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0]/num_timesteps)*num_timesteps)]
            song = np.reshape(song, [song.shape[0]//num_timesteps, song.shape[1]*num_timesteps])
            #Train the RBM on batch_size examples at a time
            for i in range(1, len(song), batch_size): 
                tr_x = song[i:i+batch_size]
                sess.run(updt, feed_dict={x: tr_x})

    sample = get_gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((50, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i,:]):
            continue
        
        S = np.reshape(sample[i,:], (num_timesteps, 2*note_range))
        midi_manipulation.noteStateMatrixToMidi(S, "music_{}".format(i+1))
            
