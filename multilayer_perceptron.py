import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score


def mlp_benchmark(X_train ,y_train ,X_test ,y_test ,hidden_layers=2,n_hidden_1=256,n_hidden_2=256 ,learning_rate=0.005 ,training_epochs=20 ,batch_size=100):

    # Create model
    def multilayer_perceptron(x,hidden_layers=hidden_layers):
        # Hidden fully connected layer with 256 neurons
        with tf.name_scope('layer_1'):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        if hidden_layers == 2:
            with tf.name_scope('layer_2'):
                layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            # Output fully connected layer with a neuron for each class
            with tf.name_scope('out_layer'):
                out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        else:
            with tf.name_scope('out_layer'):
                out_layer = tf.matmul(layer_1, weights['out']) + biases['out']

        return out_layer

    n_input=X_train.shape[1]
    n_classes = 2

    # tf Graph en entree
    with tf.name_scope('input'):
        X = tf.placeholder("float", [None, n_input])
        Y = tf.placeholder("float", [None, n_classes])

    if hidden_layers == 2:
        # poids et biais
        with tf.name_scope('weights'):
            w1=tf.Variable(tf.random_normal([n_input, n_hidden_1]))
        weights = {
            'h1': w1,
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }

        with tf.name_scope('biais'):
            b1=tf.Variable(tf.random_normal([n_hidden_1]))

        biases = {
            'b1': b1,
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        # couche cachee entierement connectee avec n_hidden_1 neurones
    else:
        # poids et biais
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
    # couche cachee entierement connectee avec n_hidden_1 neurones

    logits = multilayer_perceptron(X)
    # Fonction de cout (minimiser le cout)
    with tf.name_scope('cross_entropy'):
        diff=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)

    tf.summary.scalar('cross_entropy', cross_entropy)

    # Descente de Gradient
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(cross_entropy)

    # Test For Accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


    # Initialisation
    init = tf.global_variables_initializer()
    epoch_vs_cost = []

    # HyperParametres
    print('training_epochs : {0} learning_rate : {1} batch_size : {2}'.format(training_epochs, learning_rate, batch_size))
    display_step = 2
    total_batch = int(len(X_train) / batch_size)

    with tf.Session() as sess:

        # tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./logs/1/train ', sess.graph)
        test_writer = tf.summary.FileWriter('./logs/1/test ', sess.graph)

        sess.run(init)
        start_time = datetime.datetime.now()

        # cycle d entrainement
        for epoch in range(training_epochs):
            avg_cost = 0.

            total_batch = int(len(X_train) / batch_size)

            # boucle pour tous les mini batches
            for i in range(total_batch):

                batch_x = X_train[i * (batch_size):i * (batch_size) + batch_size]
                batch_y = y_train[i * (batch_size):i * (batch_size) + batch_size]

                batch_x_test = X_test[i * (batch_size):i * (batch_size) + batch_size]
                batch_y_test = y_test[i * (batch_size):i * (batch_size) + batch_size]

                a = y_train[i * (batch_size):i * (batch_size) + batch_size]
                batch_y = np.zeros((batch_y.shape[0], 2))
                batch_y[np.arange(batch_y.shape[0]), a] = 1

                b = y_test[i * (batch_size):i * (batch_size) + batch_size]
                batch_y_test = np.zeros((batch_y_test.shape[0], 2))
                batch_y_test[np.arange(batch_y_test.shape[0]), b] = 1

                # optimisation et backpropagation => valeurs du cout sur training
                #_, batch_cost = sess.run([train_op, cross_entropy], feed_dict={X: batch_x, Y: batch_y})
                _, batch_cost,summary = sess.run([train_op, cross_entropy,merged], feed_dict={X: batch_x, Y: batch_y})

                # Calcul de cout moyen (ajout a chaque batch)
                avg_cost += batch_cost / total_batch

                train_writer.add_summary(summary, epoch)

            if i % 10 == 0:  # Record summaries and test-set accuracy

                summary_test, acc = sess.run([merged, accuracy], feed_dict={X: batch_x_test, Y: batch_y_test})
                test_writer.add_summary(summary_test, i)
                print('Accuracy at step %s: %s' % (i, acc))

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

            epoch_vs_cost.append([epoch + 1, avg_cost])

        end_time = datetime.datetime.now()
        print("Optimization Finished!")

        #################### # Test du modele # ####################
        pred = tf.nn.softmax(logits , name="test_predictions")  # softmax en sortie
        tf.summary.histogram("test_predictions", pred)

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        incorrect_prediction = tf.not_equal(tf.argmax(pred, 1),tf.argmax(Y, 1))  # dataframes pour affichage des couts VALID VS TRAINING

        df1 = pd.DataFrame(epoch_vs_cost, columns=['epoch', 'cost'])

        with tf.name_scope('summaries_test_predictions'):
            # Calcul de l accuracy
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                tf.summary.scalar('accuracy', accuracy)
            # Calculate de l erreur
            with tf.name_scope('error'):
                error = tf.reduce_mean(tf.cast(incorrect_prediction, "float"))

        tf.summary.scalar('accuracy', accuracy)

        a=y_test
        y_test = np.zeros((y_test.shape[0], 2))
        y_test[np.arange(y_test.shape[0]), a] = 1

        print('Accuracy :{0}'.format(accuracy.eval({X: X_test, Y: y_test}, session=sess)))
        print('Erreur:{0}'.format(error.eval({X: X_test, Y: y_test}, session=sess)))

        prediction = tf.argmax(pred, 1)
        # predict
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
    # hidden_layers=2
    #n_hidden_1 = 100
    #n_hidden_2 = 100
    #training_epochs=10000
    #batch_size=50
    #learning_rate=0.00000001
    #Accuracy: 0.8471318483352661
    #Erreur: 0.1528681218624115

    # hidden_layers=2
    # n_hidden_1 = 100
    # n_hidden_2 = 100
    # training_epochs=500
    # batch_size=100
    # learning_rate=0.00000001
    # Accuracy: 0.5919574499130249
    # Erreur: 0.4080425798892975

    # 2 layers
    # hidden_layers=2
    # n_hidden_1 = 256
    # n_hidden_2 = 256
    # training_epochs=2500
    # batch_size=100
    # learning_rate=0.00000001
    # Accuracy: 0.9056771397590637
    # Erreur: 0.09432288259267807

    # 2 layers
    # hidden_layers=2
    # n_hidden_1 = 256
    # n_hidden_2 = 256
    # training_epochs=200
    # batch_size=100
    # learning_rate=0.00000001
    # Accuracy: 0.6274393796920776
    # Erreur: 0.37256062030792236

    #1 layer
    # hidden_layers=1
    # n_hidden_1 = 256
    # n_hidden_1 = 256
    # training_epochs=2500
    # batch_size=100
    # learning_rate=0.00000001
    # Accuracy: 0.5177409648895264
    # Erreur: 0.48225900530815125

    # hidden_layers=1
    # n_hidden_1 = 256
    # n_hidden_2 = 256
    # training_epochs=10000
    # batch_size=100
    # learning_rate=0.0000001
    # Accuracy: 0.8669426441192627
    # Erreur: 0.1330573558807373

    #hidden_layers=2
    #n_hidden_1 = 256
    #n_hidden_2 = 256
    #training_epochs=500
    #batch_size=100
    #learning_rate=0.00000001
    #Accuracy: 0.8914843201637268
    #Erreur: 0.1085156723856926

    hidden_layers = 2
    n_hidden_1 = 256
    n_hidden_2 = 256
    training_epochs=500
    batch_size=100
    learning_rate=0.00000001

    a = mlp_benchmark(X_train, Y_train, X_test, Y_test, hidden_layers=hidden_layers ,n_hidden_1=n_hidden_1,n_hidden_2=n_hidden_2, learning_rate=learning_rate, training_epochs=training_epochs,batch_size=batch_size)
