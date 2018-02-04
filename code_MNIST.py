Author = "SKYNET"

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os.path
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# choose a path to save the trained model_AEn
model_path = './model_AEn/'

# Download mnist Dataset from 'http://yann.lecun.com/exdb/mnist/' and put it in the specified directory
mnist = input_data.read_data_sets('../mnist_Data/', one_hot=True)

# Seperate mnist train and test data and lables and put them into ndarray files
train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels

# Network Parameters
Num_of_hLayers = 1
Num_inputs = train_data.shape[1]
Num_Outputs = len(train_labels[0])
Num_train_data = train_data.shape[0]
N_hidden1 = 60

# Parameters
Learning_rate = 0.1
Batch_size = 2
Num_batches = int(Num_train_data / Batch_size)
Num_epochs = 100

#################Defining Tensorflow Graph######################
# Graph input
In_data = tf.placeholder(dtype=tf.float32, shape=[None, Num_inputs], name="Input_layer")
Out_data = tf.placeholder(dtype=tf.float32, shape=[None, Num_Outputs], name="Output_layer")
# Weights_AEn of the network

Weights = {'W1': tf.Variable(tf.random_normal([Num_inputs, N_hidden1])),
           'W2': tf.Variable(tf.random_normal([N_hidden1, Num_Outputs]))}
Bias = {'b1': tf.Variable(tf.random_normal([N_hidden1])),
        'out': tf.Variable(tf.random_normal([Num_Outputs]))}

##################3 Defining The Model
###layer 1 is the input###
with tf.name_scope('Model'):
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(In_data, Weights['W1']), Bias['b1']))
    output_layer = tf.nn.softmax(tf.add(tf.matmul(layer_2, Weights['W2']), Bias['out']), dim=1)
    ##########################
    # it's like we have put an identity function as the activation function of the last layer
    # and then passed it to a softmax layer
    ##########################

##### Define Loss
with tf.name_scope("Loss"):
    RSME_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(Out_data - output_layer, 2), reduction_indices=1))
    cost = tf.reduce_mean(-tf.reduce_sum(Out_data * tf.log(output_layer), reduction_indices=1))

###### Define Optimizer_AEn
with tf.name_scope("Gradient_Descent_Optimizer"):
    Optimizer = tf.train.GradientDescentOptimizer(learning_rate=Learning_rate)
    train_optimizer = Optimizer.minimize(RSME_loss)
###### Define Model Accuraccy
with tf.name_scope("Accuracy"):
    Acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Out_data, 1), tf.argmax(output_layer, 1)), tf.float64))

################################ Start Training ########################
# preparing a saver_AEn to save the model_AEn
saver = tf.train.Saver()

########begining the session of training
with tf.Session() as sess:
    # first we Initialize all variables
    sess.run(tf.global_variables_initializer())

    # choose a name for the model_AEn to save it
    model_name = "MLP_MNIST_{}hidde_{}epo_{}lr".format(N_hidden1, Num_epochs, Learning_rate)
    # savig the losses and accuracies to plot later
    Train_Accs = []
    Test_Accs = []
    Train_loss = []
    Test_Loss = []
    ###EPOCHs#####
    for epoch in range(Num_epochs):
        ####check wether the model_AEn has been trained before, if it has then don't go through the training process again
        if os.path.isfile(model_path + model_name + '.meta'):
            break
        epoch_Cost = 0
        epoch_Acc = 0

        ######## training with each epoch#################

        for batch in range(Num_batches):
            batch_data = train_data[batch * Batch_size:(batch + 1) * Batch_size, :]
            batch_label = train_labels[batch * Batch_size:(batch + 1) * Batch_size, :]

            _, batch_loss, batch_accuracy = sess.run([train_optimizer, RSME_loss, Acc],
                                                     feed_dict={In_data: batch_data, Out_data: batch_label})
            epoch_Cost += batch_loss
            epoch_Acc += batch_accuracy
        epoch_Cost /= Num_batches
        epoch_Acc /= Num_batches
        # testing to see the model_AEn accuracy for the test data after each epoch
        test_loss, test_acc = sess.run([RSME_loss, Acc], feed_dict={In_data: test_data, Out_data: test_labels})

        Train_loss.append(epoch_Cost)
        Test_Accs.append(test_acc)
        Train_Accs.append(epoch_Acc)
        Test_Loss.append(test_loss)
        print("epoch number: ", epoch + 1, " Accuracy=", epoch_Acc, " Cost=", epoch_Cost)
        if (epoch + 1) == Num_epochs:
            ###at the end of the final epoch, save the model_AEn
            saver.save(sess, model_path + model_name)
            print("Model Was Saved!")

            #######Drawing plots############
            Test_Accs = [i * 100 for i in Test_Accs]
            Train_Accs = [j * 100 for j in Train_Accs]
            X_data = np.arange(Num_epochs)

            red_patch = mpatches.Patch(color='red', label='Train Accuracy')
            blue_patch = mpatches.Patch(color='blue', label='Test Accuracy')
            plt.legend(handles=[red_patch, blue_patch])
            plt.plot(X_data, Train_Accs, 'r-')
            plt.plot(X_data, Test_Accs, 'b-')
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.show()

            green_patch = mpatches.Patch(color='green', label='Train Loss')
            yellow_patch = mpatches.Patch(color='yellow', label='Test Loss')
            plt.legend(handles=[green_patch, yellow_patch])
            plt.plot(X_data, Train_loss, 'g-')
            plt.plot(X_data, Test_Loss, 'y-')
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.show()
    print("Optimization was done!")

    ###Evaluating Test Data

    saver = tf.train.import_meta_graph(model_path + model_name + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    print("Model Was Restored!")
    print("Accuracy for Test data: ")
    test_acc = sess.run(Acc, feed_dict={In_data: test_data, Out_data: test_labels})
    print("Accuracy for test data is :", test_acc * 100)

print(Author, ":)")
