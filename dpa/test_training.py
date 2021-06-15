import numpy as np
from tfdeterminism import patch
from dpa.evaluation import RepeatedAttack
import argparse
import os
import random
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from dpa.query import Model
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow_privacy.privacy.optimizers.dp_optimizer \
    import DPAdamGaussianOptimizer
from dpa.tf_priv.localDPQuery import LocalDPAdamGaussianOptimizer



if __name__ == '__main__':
    
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    eager = True
    input_shape=(28, 28, 1)
    learning_rate=0.005
    epochs=3
    iterations = 5
    repeated_attack = RepeatedAttack()
    
    m = Model()
    m.set_training_params(noise_multiplier=0,
                        learning_rate=learning_rate,
                        l2_norm_clip=None,
                        epochs=epochs,
                        delta=None)
    n_train = 100
    x_train, x_alt, y_train, y_alt, x_test, y_test = repeated_attack.get_train_test_data(bounded=True, white_picture=False, n_train=n_train)

    for i in range(iterations):
        np.random.seed(42)
        tf.random.set_seed(42)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = str(42)
        os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
        random.seed(42)
        strategy = tf.distribute.MirroredStrategy()#devices=["/gpu:0"])
        dataset = m.get_dataset(x_train, y_train)
        dataset_alt = m.get_per_example_dataset(x_alt, y_alt)
        eval_dataset = m.get_per_example_dataset(x_test, y_test)
        dataset = strategy.experimental_distribute_dataset(dataset)
        dataset_alt = strategy.experimental_distribute_dataset(dataset_alt)
        eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)
        
        with strategy.scope():
            m.reset_model()
            print("test", np.shape(m.keras_model.get_weights()))
            #print(m.keras_model.get_weights()[0])

            num_samples_orig = len(x_train)
            num_samples_alt = len(x_alt)
            optimizer = tf1.train.GradientDescentOptimizer(learning_rate)
            loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            def loss_fn(labels, predictions):
                return loss_object(labels, predictions)
        def train_step(inputs):
            images, labels = inputs
            with tf.GradientTape(persistent=True) as gradient_tape:
                var_list = m.keras_model.trainable_variables
                loss_object = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                logits = m.keras_model(images)
                #grads_tape = optimizer.compute_gradients(loss_fn(labels, logits), var_list,
                #                                gradient_tape=gradient_tape, gate_gradients=tf1.train.Optimizer.GATE_OP)
                grads_tape = gradient_tape.gradient(loss_fn(labels, logits), var_list)
            optimizer.apply_gradients(zip(grads_tape, var_list))
            return grads_tape, logits, images[0]
        @tf.function
        def distributed_train_step(dataset_inputs):
            return strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
        for epoch in range(epochs):
            for images, labels in dataset:
                grads_tape, logits, images = distributed_train_step((images, labels))
                flat_norm_from_tape = np.linalg.norm(m.flatten_tape_gradients(grads_tape))
                print("from tape ", flat_norm_from_tape)
                print("logits ", images)
                '''with tf.GradientTape(persistent=True) as gradient_tape:
                    var_list = m.keras_model.trainable_variables
                    
                    loss_object = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                    def loss_fn():
                        logits = m.keras_model(images)
                        loss = loss_object(labels, logits)
                        return loss
                    grads_tape = gradient_tape.gradient(loss_fn(), var_list)
                flat_norm_from_tape = np.linalg.norm(m.flatten_tape_gradients(grads_tape))
                print("from tape ", flat_norm_from_tape)
                optimizer.apply_gradients(zip(grads_tape, var_list))'''
    
    
    
    '''eager = True
    if eager==False:
        tf.compat.v1.disable_v2_behavior()
    input_shape=(28, 28, 1)
    learning_rate=0.005
    epochs=3
    iterations = 5
    repeated_attack = RepeatedAttack()

    n_train = 100
    x_train, x_alt, y_train, y_alt, x_test, y_test = repeated_attack.get_train_test_data(bounded=True, white_picture=False, n_train=n_train)

    for i in range(iterations):
        m = Model()
        m.set_training_params(noise_multiplier=0,
                            learning_rate=learning_rate,
                            l2_norm_clip=None,
                            epochs=epochs,
                            delta=None)
        num_samples_orig = len(x_train)
        num_samples_alt = len(x_alt)
        
        if eager:
            dataset = m.get_per_example_dataset(x_train, y_train)
            dataset_alt = m.get_per_example_dataset(x_alt, y_alt)
            eval_dataset = m.get_per_example_dataset(x_test, y_test)
            #optimizer = LocalDPAdamGaussianOptimizer(l2_norm_clip=3,
            #                                            noise_multiplier=0,
            #                                            num_microbatches=1,
            #                                            learning_rate=learning_rate,
            #                                            worst_case=False)
            optimizer = tf1.train.GradientDescentOptimizer(learning_rate)#tf.keras.optimizers.SGD(learning_rate=learning_rate)
            for epoch in range(epochs):
                for images, labels in dataset:
                    with tf.GradientTape(persistent=True) as gradient_tape:
                        var_list = m.keras_model.trainable_variables
                        loss_object = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)#tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)#CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                        def loss_fn():
                            logits = m.keras_model(images)
                            loss = loss_object(labels, logits)
                            return loss

                        #grads = optimizer.compute_gradients(loss_fn, var_list, gradient_tape=gradient_tape)
                        grads_tape = gradient_tape.gradient(loss_fn(), var_list)
                    #flat_norm_from_gradient = np.linalg.norm(m.flatten_gradients(grads))
                    #print("from dp optimizer ", flat_norm_from_gradient)
                    flat_norm_from_tape = np.linalg.norm(m.flatten_tape_gradients(grads_tape))
                    print("from tape ", flat_norm_from_tape)

                    optimizer.apply_gradients(zip(grads_tape, var_list))
        else:
            labels = tf1.placeholder(shape=(None, *y_train.shape[1:]), dtype=tf1.int32)
            features = tf1.placeholder(shape=(None,) + input_shape, dtype=tf1.float32)
            #optimizer = LocalDPAdamGaussianOptimizer(l2_norm_clip=3,
            #                                            noise_multiplier=0,
            #                                            num_microbatches=1,
            #                                            learning_rate=learning_rate,
            #                                            worst_case=False)
            #normal optimizer to compare
            normal_optimizer=tf1.train.GradientDescentOptimizer(learning_rate)
            probs = m.keras_model(features)
            loss = CategoricalCrossentropy(reduction=tf1.losses.Reduction.NONE)(labels, probs)
            loss_mean = tf1.reduce_mean(loss)
            #grads= optimizer.compute_gradients(loss, m.keras_model.trainable_weights)
            grads_normal = normal_optimizer.compute_gradients(loss_mean, m.keras_model.trainable_variables)
            update = normal_optimizer.apply_gradients(grads_normal)
            sess = tf1.keras.backend.get_session()
            Model.init_uninitialized_vars(sess)
            for epoch in range(epochs):
                for i in range(0,100):
                    #per_example_val = sess.run(grads, {labels: [y_train[i]], features: [x_train[i]]})
                    #print("example norms ", np.linalg.norm(m.undo_gradient_normalize(m.flatten_numpy_gradients(per_example_val), 1)))
                    per_example_val_normal = sess.run(grads_normal, {labels: [y_train[i]], features: [x_train[i]]})
                    print("example norms no dp", np.linalg.norm(m.undo_gradient_normalize(m.flatten_numpy_gradients(per_example_val_normal), 1)))
                    sess.run(update, {labels: [y_train[i]], features: [x_train[i]]})'''



  