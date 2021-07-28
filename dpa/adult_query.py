import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow_privacy.privacy.optimizers.dp_optimizer \
    import DPAdamGaussianOptimizer

from dpa.tf_priv.localDPQuery import LocalDPAdamGaussianOptimizer


class Model_Adult:

    def __init__(self, input_shape=(104,), classes=1):

        inputs = tf.keras.Input(shape=input_shape, name='adults')
        x1 = Dense(6, activation='relu')(inputs)
        x2 = Dense(6, activation='relu')(x1)
        outputs=Dense(classes, activation='sigmoid')(x2)

        self.keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.noise_multiplier = None
        self.learning_rate = None
        self.l2_norm_clip = None
        self.epochs = None
        self.input_shape = input_shape
        self.delta = None
    
    def get_optimizer(self,local, worst_case, num_samples_orig, num_samples_alt):

        #if local:

        baseline_optimizer = LocalDPAdamGaussianOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=0,
            num_microbatches=num_samples_orig,
            learning_rate=self.learning_rate,
            worst_case=False)

        baseline_per_example_optimizer = LocalDPAdamGaussianOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=0,
            learning_rate=self.learning_rate,
            worst_case=False)

        baseline_alt_optimizer = LocalDPAdamGaussianOptimizer(l2_norm_clip=self.l2_norm_clip,
                                                    noise_multiplier=0,
                                                    num_microbatches=num_samples_alt,
                                                    learning_rate=self.learning_rate,
                                                    worst_case=False)

        dp_optimizer = LocalDPAdamGaussianOptimizer(l2_norm_clip=self.l2_norm_clip,
                                                                noise_multiplier=self.noise_multiplier,
                                                                num_microbatches=num_samples_orig,
                                                                learning_rate=self.learning_rate,
                                                                worst_case=worst_case)

        return dp_optimizer, baseline_optimizer, baseline_alt_optimizer, baseline_per_example_optimizer

    def set_training_params(self, noise_multiplier, learning_rate,
                            l2_norm_clip, epochs, delta=None):
        self.noise_multiplier = noise_multiplier
        self.learning_rate = learning_rate
        self.l2_norm_clip = l2_norm_clip
        self.epochs = epochs
        self.delta = delta

    def flatten_numpy_gradients(self, gradients):
        flat_gradients = []
        for grad_i, _ in gradients:
            flat_gradients.append(grad_i.flatten())
        return np.concatenate(flat_gradients)

    def reshape_tails(self, tails_flat, gradients):
        tails_and_vars = []
        tailindex = 0
        for grad_i, var_i in gradients:
            #grad_shape = grad_i.shape
            grad_shape = grad_i.shape
            begin = tailindex
            end = tailindex+np.prod(grad_shape)
            tails_per_layer = tails_flat[begin:end]
            tails_per_layer = np.reshape(tails_per_layer, grad_shape)
            tails_per_layer = tf.convert_to_tensor(tails_per_layer)
            tails_and_vars.append((tails_per_layer, var_i))
            tailindex = end
        
        return tails_and_vars
    
    def train(self, train_set, alternative_set, test_set, local, worst_case, bounded):
        tf.compat.v1.disable_v2_behavior()
        x_test, y_test = test_set
        x_train, y_train = train_set
        x_alt, y_alt = alternative_set
        num_samples_orig = len(x_train)
        num_samples_alt = len(x_alt)
        print("samples orig ", num_samples_orig, " samples alt ", num_samples_alt, " y train shape ", y_train.shape)
        print(y_train)

        labels = tf1.placeholder(shape=(None, *y_train.shape[1:]), dtype=tf1.int32)
        features = tf1.placeholder(shape=(None,) + self.input_shape, dtype=tf1.float32)
        priv_optimizer, optimizer, optimizer_alt, per_example_optimizer = self.get_optimizer(local=local,
                                                        worst_case=worst_case,
                                                        num_samples_orig = num_samples_orig, 
                                                        num_samples_alt = num_samples_alt)

        # create update and gradient tensors
        probs = self.keras_model(features)
        loss = tf.keras.losses.BinaryCrossentropy(reduction=tf1.losses.Reduction.NONE)(labels, probs)
        loss_mean = tf1.reduce_mean(loss)
        grads = optimizer.compute_gradients(loss, self.keras_model.trainable_weights)
        grads_alt = optimizer_alt.compute_gradients(loss, self.keras_model.trainable_weights)
        per_example_grads = per_example_optimizer.compute_gradients(loss, self.keras_model.trainable_weights)


        # create accuracy tensors
        test_x_placeholder = tf1.placeholder(shape=(None,) + self.input_shape, dtype=tf.float32)
        test_acc_metric = tf1.keras.metrics.BinaryAccuracy()
        test_logits = self.keras_model(test_x_placeholder)
        calc_test_acc = test_acc_metric.update_state(y_test, test_logits)
        test_acc = test_acc_metric.result()

        accuracy = tf1.keras.metrics.BinaryAccuracy()
        update_acc = accuracy.update_state(labels, probs)
        get_acc = accuracy.result()

        sess = tf1.keras.backend.get_session()
        Model_Adult.init_uninitialized_vars(sess)

        # lists to store gradient information
        original_grad_list = []
        alt_grad_list = []
        private_grad_list = []
        train_acc_list=[]
        test_acc_list=[]
        max_gradient_norm =0
        sum_gradient_norm=0

        for epoch in range(self.epochs):
            alt_grad_val = sess.run(grads_alt, feed_dict={labels: y_alt, features: x_alt, test_x_placeholder: x_test})

            feed = {labels: y_train, features: x_train, test_x_placeholder: x_test}
            fetches = [grads, loss_mean, update_acc, calc_test_acc]
            grad_val, loss_val, _ , test_acc_calc= sess.run(fetches, feed)

            #for i in range(0,100):
            #    per_example_val = sess.run(per_example_grads, {labels: [y_train[i]], features: [x_train[i]], test_x_placeholder: x_test})
                #uncomment to print all per example gradients
            #    example_norms = np.linalg.norm(self.undo_gradient_normalize(self.flatten_numpy_gradients(per_example_val), 1))
            #    if (example_norms-3)>0.1:
            #        print("example norms ", example_norms)
                #for norm in example_norms:
                #    if (norm-3)>0.1:
                #        print("example norms", norm)
                #print("example norms ", [if (norm-3)>0.1 norm for norm in example_norms])


            flat_original_grads = self.flatten_numpy_gradients(grad_val)
            flat_original_grads = self.undo_gradient_normalize(flat_original_grads, num_samples_orig)
            flat_alt_grads = self.flatten_numpy_gradients(alt_grad_val)
            flat_alt_grads = self.undo_gradient_normalize(flat_alt_grads, num_samples_alt)

            #if np.linalg.norm(flat_alt_grads)>max_gradient_norm:
            #    max_gradient_norm = np.linalg.norm(flat_alt_grads)
            if np.linalg.norm(flat_original_grads)>max_gradient_norm:
                max_gradient_norm=np.linalg.norm(flat_original_grads)
            sum_gradient_norm = sum_gradient_norm + np.linalg.norm(flat_original_grads)

            current_sensitivity = self.compute_gradient_norm(flat_original_grads, flat_alt_grads)
            print("current sensitivity ", current_sensitivity, flush=True)
            print("l2_norm ", self.l2_norm_clip, flush=True)

            if local:
                current_sensitivity = self.compute_gradient_norm(flat_original_grads, flat_alt_grads)
                priv_optimizer.set_sensitivity(current_sensitivity)
                if worst_case:
                    tails_flat = self.worst_case_noise(flat_gradients=flat_original_grads,
                                                       flat_alt_grad=flat_alt_grads,
                                                       local_sensitivity=current_sensitivity)
                    tails_and_vars = self.reshape_tails(tails_flat=tails_flat,
                                                        gradients=grad_val)
                    tails = [tail for tail, _ in tails_and_vars]
                    priv_optimizer.set_tail(tails)

            elif bounded:
                priv_optimizer.set_sensitivity(2*self.l2_norm_clip)


            private_grads_op = priv_optimizer.compute_gradients(loss,
                                                                self.keras_model.trainable_weights)
            print('sensitivity adding noise ', priv_optimizer.get_sensitivity(), flush=True)
            # print current gradient norm
            print(self.compute_gradient_norm(flat_original_grads, flat_alt_grads), flush=True)
            print("avg gradient norm ", np.linalg.norm(flat_original_grads), flush=True)

            update = priv_optimizer.apply_gradients(private_grads_op)
            Model_Adult.init_uninitialized_vars(sess)

            private_grads = sess.run(private_grads_op, feed_dict=feed)

            accuracy_val, test_accuracy_val, _ = sess.run([get_acc, test_acc, update], feed_dict=feed)
            accuracy.reset_states()
            test_acc_metric.reset_states()

            flat_private_grads = self.flatten_numpy_gradients(private_grads)
            flat_private_grads = self.undo_gradient_normalize(flat_private_grads, num_samples_orig)

            original_grad_list.append(flat_original_grads)
            alt_grad_list.append(flat_alt_grads)
            private_grad_list.append(flat_private_grads)
            train_acc_list.append(accuracy_val)
            test_acc_list.append(test_accuracy_val)
            print("magnitude nosie added ", self.compute_gradient_norm(flat_original_grads, flat_private_grads), flush=True)

            if (epoch + 1) % 1 == 0:
                print("Epoch: ", epoch + 1, " Loss: ", loss_val, " Train Accuracy: ", accuracy_val, " Test Accuracy: ", test_accuracy_val, flush=True)

        print("max gradient norm ", max_gradient_norm, flush=True)
        return original_grad_list, alt_grad_list, private_grad_list, train_acc_list, test_acc_list

    def worst_case_noise(self, flat_gradients, flat_alt_grad, local_sensitivity):
        delta_i = self.delta / self.epochs
        epsilon_i = np.sqrt(2 * np.log(1.25 / delta_i)) / self.noise_multiplier

        part1 = epsilon_i * (local_sensitivity * self.noise_multiplier) ** 2 / local_sensitivity
        part2 = local_sensitivity / 2
        magnitude = part1 - part2

        direction = (flat_gradients - flat_alt_grad) / local_sensitivity
        return direction * magnitude

    def undo_gradient_normalize(self, gradients, batch_size):
        return np.multiply(gradients, batch_size)

    def compute_gradient_norm(self, orig_grads, alt_grads):
        return np.linalg.norm(orig_grads - alt_grads)

    @staticmethod
    def init_uninitialized_vars(sess):
        uninitialized_vars = []
        for var in tf1.all_variables():
            try:
                sess.run(var)
            except tf1.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        init_new_vars_op = tf1.initialize_variables(uninitialized_vars)
        sess.run(init_new_vars_op)

    def save(self, path):
        self.keras_model.save(path)