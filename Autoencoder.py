Author = "SKYNET"

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os.path
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# choose a path to save the trained model_AEn and model_MLP
model_path_AEn = './model_AEn/'
model_path_MLP = './model_MLP/'

# Download mnist Dataset from 'http://yann.lecun.com/exdb/mnist/' and put it in the specified directory
mnist = input_data.read_data_sets('../mnist_Data/', one_hot=True)

# Seperate mnist train and test data and lables and put them into ndarray files
train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels

#################### Network Parameters
######AutoEncoder##############
Num_of_AEn_hLayers = 1
Num_AEn_inputs = train_data.shape[1]
Num_AEn_Outputs = train_data.shape[1]
N_hidden1_AEn = 350

##########MLP#####################
Num_of_MLP_hLayers = 1
Num_MLP_inputs = N_hidden1_AEn
Num_MLP_Outputs = len(train_labels[0])
N_hidden1_MLP = 60

# Parameters
Learning_rate_AEn = 0.01
Learning_rate_MLP = 0.001
Batch_size = 20
Num_train_data = train_data.shape[0]
Num_batches = int(Num_train_data / Batch_size)
Num_epochs_AEn = 100
Num_epochs_MLP = 100
closeness = 0.5

#################Defining Tensorflow Graph######################
# Graph input
In_data = tf.placeholder(dtype=tf.float32, shape=[None, Num_AEn_inputs], name="Input_layer")
AEn_Exp_Outdata = tf.placeholder(dtype=tf.float32, shape=[None, Num_AEn_Outputs], name="Output_layer_Autoencoder")
MLP_Exp_Outdata = tf.placeholder(dtype=tf.float32, shape=[None, Num_MLP_Outputs], name="Output_layer_MLP")
#AEn_to_MLP_Weights = tf.placeholder(dtype=tf.float32, shape=[Num_AEn_inputs, N_hidden1_AEn])
#AEn_to_MLP_Bias = tf.placeholder(dtype=tf.float32, shape=[N_hidden1_AEn])
# Weights_AEn of the network

Weights_AEn = {'W1': tf.Variable(tf.random_normal([Num_AEn_inputs, N_hidden1_AEn])),
               'W2': tf.Variable(tf.random_normal([N_hidden1_AEn, Num_AEn_Outputs]))}
Bias_AEn = {'b1': tf.Variable(tf.random_normal([N_hidden1_AEn])),
            'out': tf.Variable(tf.random_normal([Num_AEn_Outputs]))}

Weights_MLP = {'W1': tf.Variable(tf.random_normal([Num_MLP_inputs, N_hidden1_MLP])),
               'W2': tf.Variable(tf.random_normal([N_hidden1_MLP, Num_MLP_Outputs]))}
Bias_MLP = {'b1': tf.Variable(tf.random_normal([N_hidden1_MLP])),
            'out': tf.Variable(tf.random_normal([Num_MLP_Outputs]))}

##################3 Defining The Model
###layer 1 is the input###
with tf.name_scope('Model_AutoEncoder'):
    AEn_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(In_data, Weights_AEn['W1']), Bias_AEn['b1']))
    AEn_Est_output = tf.nn.sigmoid(tf.add(tf.matmul(AEn_layer_2, Weights_AEn['W2']), Bias_AEn['out']))
    ##########################
    # it's like we have put an identity function as the activation function of the last layer
    # and then passed it to a softmax layer
    ##########################
with tf.name_scope('Model_MLP'):
    MLP_layer1 = tf.nn.tanh(tf.add(tf.matmul(In_data, Weights_AEn['W1']), Bias_AEn['b1']))
    MLP_layer_2 = tf.nn.tanh(tf.add(tf.matmul(MLP_layer1, Weights_MLP['W1']), Bias_MLP['b1']))
    #MLP_Est_output = tf.nn.softmax(tf.add(tf.matmul(MLP_layer_2, Weights_MLP['W2']), Bias_MLP['out']), dim=1)
    MLP_Est_output = tf.add(tf.matmul(MLP_layer_2, Weights_MLP['W2']), Bias_MLP['out'])
##### Define Loss#################
with tf.name_scope("Loss_AutoEncoder"):
    RSME_loss_AEn = tf.reduce_mean(tf.reduce_sum(tf.pow(AEn_Exp_Outdata - AEn_Est_output, 2), reduction_indices=1))

with tf.name_scope("Loss_MLP"):
    #RSME_loss_MLP = tf.reduce_mean(tf.reduce_sum(tf.pow(MLP_Exp_Outdata - MLP_Est_output, 2), reduction_indices=1))
    RSME_loss_MLP =tf.losses.softmax_cross_entropy(logits=MLP_Est_output,
                                               onehot_labels=MLP_Exp_Outdata)
###### Define Optimizer_AEn
with tf.name_scope("Gradient_Descent_Optimizer"):
    Optimizer_AEn = tf.train.GradientDescentOptimizer(learning_rate=Learning_rate_AEn)
    train_optimizer_AEn = Optimizer_AEn.minimize(RSME_loss_AEn)

    Optimizer_MLP = tf.train.GradientDescentOptimizer(learning_rate=Learning_rate_AEn)
    train_optimizer_MLP = Optimizer_MLP.minimize(RSME_loss_MLP,var_list=[Weights_MLP['W1'],Weights_MLP['W2'],Bias_MLP['b1'],Bias_MLP['out']])

###### Define Model Accuraccy
with tf.name_scope("Accuracy"):
    Acc_MLP = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(MLP_Exp_Outdata, 1), tf.argmax(MLP_Est_output, 1)), tf.float64))
    Acc_AEn = tf.reduce_mean(tf.cast(
        tf.less(tf.sqrt(tf.reduce_sum(tf.pow(AEn_Exp_Outdata - AEn_Est_output, 2), reduction_indices=1)), closeness),
        tf.float64))
################################ Start Training ########################
# preparing a saver_AEn to save the model_AEn
saver_AEn = tf.train.Saver()
saver_MLP = tf.train.Saver()

########begining the session of training
with tf.Session() as sess:
    # first we Initialize all variables
    sess.run(tf.global_variables_initializer())

    # choose a name for the model_AEn to save it
    AEn_model_name = "AEn_MNIST_{}hidden_{}epo_{}lr".format(N_hidden1_AEn, Num_epochs_AEn, Learning_rate_AEn)
    # savig the losses and accuracies to plot later
    Train_Accs_AEn = []
    Test_Accs_AEn = []
    Train_loss_AEn = []
    Test_Loss_AEn = []
    ###EPOCHs#####
    print("Training of The Autoencoder Started!")
    for epoch in range(Num_epochs_AEn):
        ####check wether the model_AEn has been trained before, if it has then don't go through the training process again
        if os.path.isfile(model_path_AEn + AEn_model_name + '.meta'):
            break
        epoch_Cost = 0
        epoch_Acc = 0

        ######## training with each epoch#################

        for batch in range(Num_batches):
            batch_data = train_data[batch * Batch_size:(batch + 1) * Batch_size, :]

            _, batch_loss, batch_accuracy = sess.run([train_optimizer_AEn, RSME_loss_AEn, Acc_AEn],
                                                     feed_dict={In_data: batch_data, AEn_Exp_Outdata: batch_data})
            epoch_Cost += batch_loss
            epoch_Acc += batch_accuracy
        epoch_Cost /= Num_batches
        epoch_Acc /= Num_batches
        # testing to see the model_AEn accuracy for the test data after each epoch
        test_loss, test_acc = sess.run([RSME_loss_AEn, Acc_AEn],
                                       feed_dict={In_data: test_data, AEn_Exp_Outdata: test_data})

        Train_loss_AEn.append(epoch_Cost)
        Test_Accs_AEn.append(test_acc)
        Train_Accs_AEn.append(epoch_Acc)
        Test_Loss_AEn.append(test_loss)
        print("epoch number: ", epoch + 1, " Accuracy=", epoch_Acc, " Cost=", epoch_Cost)
        if (epoch + 1) == Num_epochs_AEn:
            ###at the end of the final epoch, save the model_AEn
            saver_AEn.save(sess, model_path_AEn + AEn_model_name)
            print("Autoencoder Model Was Saved !")

            #######Drawing plots############
            Test_Accs_AEn = [i * 100 for i in Test_Accs_AEn]
            Train_Accs_AEn = [j * 100 for j in Train_Accs_AEn]
            X_data = np.arange(Num_epochs_AEn)

            red_patch = mpatches.Patch(color='red', label='Train Accuracy')
            blue_patch = mpatches.Patch(color='blue', label='Test Accuracy')
            plt.legend(handles=[red_patch, blue_patch])
            plt.plot(X_data, Train_Accs_AEn, 'r-')
            plt.plot(X_data, Test_Accs_AEn, 'b-')
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.show()

            green_patch = mpatches.Patch(color='green', label='Train Loss')
            yellow_patch = mpatches.Patch(color='yellow', label='Test Loss')
            plt.legend(handles=[green_patch, yellow_patch])
            plt.plot(X_data, Train_loss_AEn, 'g-')
            plt.plot(X_data, Test_Loss_AEn, 'y-')
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.show()
    print("Optimization for Autoencoder was done!")

    ###Evaluating Test Data for Autoencoder Model

    saver_AEn = tf.train.import_meta_graph(model_path_AEn + AEn_model_name + '.meta')
    saver_AEn.restore(sess, tf.train.latest_checkpoint(model_path_AEn))
    print("Model Was Restored!")
    print("Accuracy for Test data: ")
    test_acc = sess.run(Acc_AEn, feed_dict={In_data: test_data, AEn_Exp_Outdata: test_data})
    print("Accuracy for test data is :", test_acc * 100)


    ################Training MLP Model from Autoencoders hidden layer#########################################

    # choose a name for the model_MLP to save it
    MLP_model_name = "MLP_MNIST_{}hidden_{}epo_{}lr".format(N_hidden1_MLP, Num_epochs_MLP, Learning_rate_MLP)
    # savig the losses and accuracies to plot later
    Train_Accs_MLP = []
    Test_Accs_MLP = []
    Train_loss_MLP = []
    Test_Loss_MLP = []
    ###EPOCHs#####
    print("Training of The MLP Started !")
    for epoch in range(Num_epochs_MLP):
        ####check wether the model_MLP has been trained before, if it has then don't go through the training process again
        if os.path.isfile(model_path_MLP + MLP_model_name + '.meta'):
            break
        epoch_Cost = 0
        epoch_Acc = 0

        ######## training with each epoch#################

        for batch in range(Num_batches):
            batch_data = train_data[batch * Batch_size:(batch + 1) * Batch_size, :]
            batch_label = train_labels[batch * Batch_size:(batch + 1) * Batch_size, :]

            _, batch_loss, batch_accuracy = sess.run([train_optimizer_MLP, RSME_loss_MLP, Acc_MLP],
                                                     feed_dict={In_data: batch_data, MLP_Exp_Outdata: batch_label})# ,AEn_to_MLP_Weights: Weights_AEn['W1'],AEn_to_MLP_Bias: Bias_AEn['b1']
            epoch_Cost += batch_loss
            epoch_Acc += batch_accuracy
        epoch_Cost /= Num_batches
        epoch_Acc /= Num_batches
        # testing to see the model_MLP accuracy for the test data after each epoch
        test_loss, test_acc = sess.run([RSME_loss_MLP, Acc_MLP],
                                       feed_dict={In_data: test_data, MLP_Exp_Outdata: test_labels})

        Train_loss_MLP.append(epoch_Cost)
        Test_Accs_MLP.append(test_acc)
        Train_Accs_MLP.append(epoch_Acc)
        Test_Loss_MLP.append(test_loss)
        print("epoch number: ", epoch + 1, " Accuracy=", epoch_Acc, " Cost=", epoch_Cost)
        if (epoch + 1) == Num_epochs_MLP:
            ###at the end of the final epoch, save the model_MLP
            saver_MLP.save(sess, model_path_MLP + MLP_model_name)
            print("MLP Model Was Saved !")

            #######Drawing plots############
            Test_Accs_MLP = [i * 100 for i in Test_Accs_MLP]
            Train_Accs_MLP = [j * 100 for j in Train_Accs_MLP]
            X_data = np.arange(Num_epochs_MLP)

            red_patch = mpatches.Patch(color='red', label='Train Accuracy')
            blue_patch = mpatches.Patch(color='blue', label='Test Accuracy')
            plt.legend(handles=[red_patch, blue_patch])
            plt.plot(X_data, Train_Accs_MLP, 'r-')
            plt.plot(X_data, Test_Accs_MLP, 'b-')
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.show()

            green_patch = mpatches.Patch(color='green', label='Train Loss')
            yellow_patch = mpatches.Patch(color='yellow', label='Test Loss')
            plt.legend(handles=[green_patch, yellow_patch])
            plt.plot(X_data, Train_loss_MLP, 'g-')
            plt.plot(X_data, Test_Loss_MLP, 'y-')
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.show()
    print("Optimization for MLP was done!")

    ###Evaluating Test Data for MLP Model

    saver_MLP = tf.train.import_meta_graph(model_path_MLP + MLP_model_name + '.meta')
    saver_MLP.restore(sess, tf.train.latest_checkpoint(model_path_MLP))
    print("MLP Model Was Restored!")
    print("Accuracy for Test data: ")
    test_acc = sess.run(Acc_MLP, feed_dict={In_data: test_data, MLP_Exp_Outdata: test_labels})
    print("Accuracy for test data is :", test_acc * 100)

print(Author, ":)")
