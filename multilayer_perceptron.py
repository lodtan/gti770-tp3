import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

def mlp_benchmark(X_train ,y_train ,X_test ,y_test ,n_hidden_1=10 ,learning_rate=0.005 ,training_epochs=20 ,batch_size=100):

    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with n_hidden_1 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    n_hidden_1=n_hidden_1
    n_input=X_train.shape[1]
    n_classes = 2

    # tf Graph en entree
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])  # poids et biais

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))}

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])), 'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # couche cachee entierement connectee avec n_hidden_1 neurones
    print(weights['h1'])

    logits = multilayer_perceptron(X)

    # Reformattage des vecteurs au format tensorflow
    def y_vector(index):
        zero = np.zeros(n_classes)
        zero[index] = 1
        return zero

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
            total_batch = int(len(X_train) / batch_size)

            # boucle pour tous les mini batches
            for i in range(total_batch):

                batch_x = X_train[i * (batch_size):i * (batch_size) + batch_size]
                batch_y = [y_vector(x) for x in y_train[i * (batch_size):i * (batch_size) + batch_size]]

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
        incorrect_prediction = tf.not_equal(tf.argmax(pred, 1),
                                            tf.argmax(Y, 1))  # dataframes pour affichage des couts VALID VS TRAINING

        df1 = pd.DataFrame(epoch_vs_cost, columns=['epoch', 'cost'])

        # Calcul de l accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # Calculate de l erreur
        error = tf.reduce_mean(tf.cast(incorrect_prediction, "float"))

        print('Accuracy :{0}'.format(accuracy.eval({X: X_test, Y: [y_vector(x) for x in y_test]}, session=sess)))
        print('Erreur:{0}'.format(error.eval({X: X_test, Y: [y_vector(x) for x in y_test]}, session=sess)))

        prediction = tf.argmax(pred, 1)
        test_y = [y_vector(x) for x in y_test]
        y_test_hat = prediction.eval(feed_dict={X: X_test}, session=sess)
        print('Predictions : {0}'.format(prediction.eval(feed_dict={X: X_test}, session=sess)))

        return {'accuracy': accuracy.eval({X: X_test, Y: [y_vector(x) for x in y_test]}),
                'error': error.eval({X: X_test, Y: [y_vector(x) for x in y_test]}),
                'y_hat': y_test_hat,
                'df_epoch': df1,
                'total_time': (end_time - start_time).total_seconds()}


if __name__ == "__main__":

    feature_vectors = pd.read_csv('galaxy_feature_vectors.csv', delimiter=',', header=None)
    X_galaxy = pd.read_csv('galaxy_feature_vectors.csv', delimiter=',', header=None).values[:, 0:-1]
    Y_galaxy = pd.read_csv('galaxy_feature_vectors.csv', delimiter=',', header=None).values[:, -1:].astype(int).flatten()
    X_train, X_test, Y_train, Y_test = train_test_split(X_galaxy, Y_galaxy, test_size=0.20, random_state=42,stratify=Y_galaxy)

    n_input = X_train.shape[1]
    n_classes = 2
    n_hidden_1 = 20

    # tf Graph en entree
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])  # poids et biais

    a = mlp_benchmark(X_train, Y_train, X_test, Y_test, n_hidden_1=n_hidden_1, learning_rate=0.1, training_epochs=2,batch_size=100)
