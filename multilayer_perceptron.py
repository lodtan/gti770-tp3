import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

def mlp_benchmark(X_train ,y_train ,X_test ,y_test ,n_hidden_1=256,n_hidden_2=256 ,learning_rate=0.005 ,training_epochs=20 ,batch_size=100,mnist_data=True):

    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    if mnist_data:

        # Import MNIST data
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        n_input = 784  # MNIST data input (img shape: 28*28)
        n_classes = 10  # MNIST total classes (0-9 digits)

    else:
        n_input=X_train.shape[1]
        n_classes = 2

    # tf Graph en entree
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])  # poids et biais

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # couche cachee entierement connectee avec n_hidden_1 neurones
    print(weights['h1'])

    logits = multilayer_perceptron(X)

    # Fonction de cout (minimiser le cout)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    # Descente de Gradient
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op)

    # Initialisation
    init = tf.global_variables_initializer()
    epoch_vs_cost = []

    # HyperParametres
    print('training_epochs : {0} learning_rate : {1} batch_size : {2}'.format(training_epochs, learning_rate, batch_size))
    display_step = 2
    total_batch = int(len(X_train) / batch_size)

    with tf.Session() as sess:

        sess.run(init)
        start_time = datetime.datetime.now()

        # cycle d entrainement
        for epoch in range(training_epochs):
            avg_cost = 0.

            if mnist_data:
                total_batch = int(mnist.train.num_examples / batch_size)
            else:
                total_batch = int(len(X_train) / batch_size)

            # boucle pour tous les mini batches
            for i in range(total_batch):

                if mnist_data:

                    batch_x, batch_y = mnist.train.next_batch(batch_size)

                else:

                    batch_x = X_train[i * (batch_size):i * (batch_size) + batch_size]
                    batch_y = y_train[i * (batch_size):i * (batch_size) + batch_size]
                    a = y_train[i * (batch_size):i * (batch_size) + batch_size]
                    batch_y = np.zeros((batch_y.shape[0], 2))
                    batch_y[np.arange(batch_y.shape[0]), a] = 1

                # optimisation et backpropagation => valeurs du cout sur training
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

                # Calcul de cout moyen (ajout a chaque batch)
                avg_cost += c / total_batch

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

            epoch_vs_cost.append([epoch + 1, avg_cost])

        end_time = datetime.datetime.now()
        print("Optimization Finished!")

        #################### # Test du modele # ####################
        pred = tf.nn.softmax(logits)  # softmax en sortie
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        incorrect_prediction = tf.not_equal(tf.argmax(pred, 1),tf.argmax(Y, 1))  # dataframes pour affichage des couts VALID VS TRAINING

        df1 = pd.DataFrame(epoch_vs_cost, columns=['epoch', 'cost'])

        # Calcul de l accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # Calculate de l erreur
        error = tf.reduce_mean(tf.cast(incorrect_prediction, "float"))

        if mnist_data:

            X_test = mnist.test.images
            y_test = mnist.test.labels

        else:

            a=y_test
            y_test = np.zeros((y_test.shape[0], 2))
            y_test[np.arange(y_test.shape[0]), a] = 1

        print('Accuracy :{0}'.format(accuracy.eval({X: X_test, Y: y_test}, session=sess)))
        print('Erreur:{0}'.format(error.eval({X: X_test, Y: y_test}, session=sess)))

        prediction = tf.argmax(pred, 1)
        y_test_hat = prediction.eval(feed_dict={X: X_test}, session=sess)
        print('Predictions : {0}'.format(prediction.eval(feed_dict={X: X_test}, session=sess)))

        return {'accuracy': accuracy.eval({X: X_test, Y: y_test}),
                'error': error.eval({X: X_test, Y: y_test}),
                'y_hat': y_test_hat,
                'df_epoch': df1,
                'total_time': (end_time - start_time).total_seconds()}


if __name__ == "__main__":

    feature_vectors = pd.read_csv('galaxy_feature_vectors.csv', delimiter=',', header=None)
    X_galaxy = pd.read_csv('galaxy_feature_vectors.csv', delimiter=',', header=None).values[:, 0:-1]
    Y_galaxy = pd.read_csv('galaxy_feature_vectors.csv', delimiter=',', header=None).values[:, -1:].astype(int).flatten()
    X_train, X_test, Y_train, Y_test = train_test_split(X_galaxy, Y_galaxy, test_size=0.20, random_state=42,stratify=Y_galaxy)

    # 2 layers
    #n_hidden_1 = 100
    #n_hidden_2 = 100
    #training_epochs=10000
    #batch_size=50
    #learning_rate=0.00000001
    #Accuracy: 0.8471318483352661
    #Erreur: 0.1528681218624115

    # n_hidden_1 = 100
    # n_hidden_2 = 100
    # training_epochs=500
    # batch_size=100
    # learning_rate=0.00000001
    # Accuracy: 0.5919574499130249
    # Erreur: 0.4080425798892975

    n_hidden_1 = 256
    n_hidden_2 = 256
    training_epochs=10000
    batch_size=50
    learning_rate=0.00000001


    #mnist_data = True
    mnist_data = False

    a = mlp_benchmark(X_train, Y_train, X_test, Y_test, n_hidden_1=n_hidden_1,n_hidden_2=n_hidden_2, learning_rate=learning_rate, training_epochs=training_epochs,batch_size=batch_size,mnist_data=mnist_data)
