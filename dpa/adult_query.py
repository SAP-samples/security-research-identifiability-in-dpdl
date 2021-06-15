import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Flatten
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow_privacy.privacy.optimizers.dp_optimizer \
    import DPAdamGaussianOptimizer

from dpa.tf_priv.localDPQuery import LocalDPAdamGaussianOptimizer


class Model_Adult:

    def __init__(self, input_shape=(105,), classes=1):

        inputs = tf.keras.Input(shape=input_shape, name='digits')
        x = Dense(6, activation='relu', input_shape=input_shape)(inputs)
        outputs=Dense(classes, activation='sigmoid')(x)

        self.keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.noise_multiplier = None
        self.learning_rate = None
        self.l2_norm_clip = None
        self.epochs = None
        self.input_shape = input_shape
        self.delta = None

    def reset_model(self):
        input_shape=(105,)
        classes=1
        inputs = tf.keras.Input(shape=input_shape, name='digits')
        x = Dense(6, activation='relu', input_shape=input_shape)(inputs)
        outputs=Dense(classes, activation='sigmoid')(x)

        self.keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
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

    def flatten_gradients(self, gradients):
        flat_gradients = []
        flat_gradients1=[]
        for grad_i, _ in gradients:
            flat_gradients.append(grad_i.numpy().flatten())
            #print("per replica norm ", grad_i.numpy())
        return np.concatenate(flat_gradients)
    
    def flatten_gradients_parallel(self, gradients):
        flat_gradients = []
        for grad_i, _ in gradients:
            #flat_gradients.append(grad_i.numpy().flatten())
            #grads_both_replicas = np.array([])
            #for num in grad_i.values[0].numpy():
            #    np.concatenate((grads_both_replicas, num))
            #for num in grad_i.values[1].numpy():
            #    np.concatenate((grads_both_replicas, num))
            #print("per replica norm ", grad_i.values[0].numpy())
            grads_sum_replicas = grad_i.values[0].numpy().flatten()
            for i in range(len(grad_i.values)):
                grads_sum_replicas = np.add(grads_sum_replicas, grad_i.values[i+1].numpy().flatten())
            #grads_both_replicas = np.add(grad_i.values[0].numpy().flatten(), grad_i.values[1].numpy().flatten())/2
            grads_sum_replicas = grads_sum_replicas/len(grad_i.values)

            #print("combined ", grads_both_replicas)
            flat_gradients.append(grads_sum_replicas)
        #print("grad normal", flat_gradients[1][0])
        #print("new grad", flat_gradients1[1][0])
        return np.concatenate(flat_gradients)

    
    def flatten_tape_gradients(self, gradients):
        flat_gradients = []
        for grad_i in gradients:
            flat_gradients.append(grad_i.numpy().flatten())
        return np.concatenate(flat_gradients)

    def flatten_numpy_gradients(self, gradients):
        flat_gradients = []
        for grad_i, _ in gradients:
            flat_gradients.append(grad_i.flatten())
        return np.concatenate(flat_gradients)

    def flatten_per_example_gradients(self, gradients):
        flat_gradients = []
        for grad_i in gradients:
            #print("gradient shape ", np.shape(grad_i))
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

    def get_dataset(self, data, labels):
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(data, tf.float32),
            tf.cast(labels, tf.int64)))
        dataset = dataset.shuffle(1000, seed=42, reshuffle_each_iteration=False).batch(len(data))
        return dataset
    def get_per_example_dataset(self, data, labels):
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(data, tf.float32),
            tf.cast(labels, tf.int64)))
        dataset = dataset.shuffle(1000, seed=42, reshuffle_each_iteration=False).batch(1)
        return dataset
    
    def train_eager(self, train_set, alternative_set, test_set, local, worst_case, bounded):
        x_test, y_test = test_set
        x_train, y_train = train_set
        x_alt, y_alt = alternative_set
        num_samples_orig = len(x_train)
        num_samples_alt = len(x_alt)
        dataset = self.get_dataset(x_train, y_train)
        dataset_alt = self.get_dataset(x_alt, y_alt)
        eval_dataset = self.get_dataset(x_test, y_test)
        strategy = tf.distribute.MirroredStrategy()#devices=["/gpu:0"])
        dataset = strategy.experimental_distribute_dataset(dataset)
        dataset_alt = strategy.experimental_distribute_dataset(dataset_alt)
        eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)

        with strategy.scope():
            self.reset_model()
            priv_opt, opt, alt_opt, _ = self.get_optimizer(local=local, 
                worst_case=worst_case, 
                num_samples_orig = num_samples_orig,
                num_samples_alt=num_samples_alt)
            test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
            train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
            def priv_train_step(original):
                images, labels = original
                with tf.GradientTape(persistent=True) as gradient_tape:
                    var_list = self.keras_model.trainable_variables
                    loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                    def loss_fn():
                        logits = self.keras_model(images)
                        loss = loss_object(labels, logits)
                        return loss                
                    priv_grads = priv_opt.compute_gradients(loss_fn, var_list, gradient_tape=gradient_tape)
                priv_opt.apply_gradients(priv_grads)
                return priv_grads
            def train_step(original, alt):
                images, labels = original
                images_alt, labels_alt = alt

                with tf.GradientTape(persistent=True) as gradient_tape:
                    logits_alt = self.keras_model(images_alt)
                    var_list = self.keras_model.trainable_variables
                    loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                    def loss_fn_alt():
                        logits = self.keras_model(images_alt)
                        loss = loss_object(labels_alt, logits)
                        return loss
                    def loss_fn():
                        logits = self.keras_model(images)
                        loss = loss_object(labels, logits)
                        return loss
                    scalar_loss = tf.reduce_mean(loss_fn())
                    grads_and_vars = opt.compute_gradients(loss_fn, var_list,
                                                gradient_tape=gradient_tape)
                    grads_alt = alt_opt.compute_gradients(loss_fn_alt, var_list, gradient_tape=gradient_tape)
                return grads_and_vars, grads_alt, scalar_loss
            def get_test_acc(test_dataset):
                images, labels = test_dataset
                logits = self.keras_model(images)
                calc_test_acc = test_acc_metric.update_state(labels, logits)
            def get_train_acc(train_dataset):
                images, labels = train_dataset
                logits = self.keras_model(images)
                calc_train_acc = train_acc_metric.update_state(labels, logits)

            
            original_grad_list = []
            alt_grad_list = []
            private_grad_list = []
            train_acc_list =[]
            test_acc_list = []
            #test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

            for epoch in range(self.epochs):
                for (images, labels), (images_alt, labels_alt), (images_test, labels_test) in zip(dataset, dataset_alt, eval_dataset):#zip(enumerate(dataset.take(-1)), enumerate(dataset_alt.take(-1))):
                    @tf.function
                    def distributed_grad(set1):
                        per_replica_original = strategy.experimental_run_v2(priv_train_step, args=(set1,))
                        return per_replica_original
                    @tf.function
                    def distributed_train_step(set1, alt_set):
                        o_grads, a_grads, loss = strategy.experimental_run_v2(train_step, args=(set1, alt_set,))
                        return o_grads, a_grads, loss
                    @tf.function
                    def get_test_acc_tf(set1):
                        accuracy = strategy.experimental_run_v2(get_test_acc, args=(set1,))
                        return accuracy
                    @tf.function
                    def get_train_acc_tf(set1):
                        accuracy = strategy.experimental_run_v2(get_train_acc, args=(set1,))
                        return accuracy
                    o_grads, a_grads, loss = distributed_train_step((images, labels), (images_alt, labels_alt))
                    if (epoch + 1) % 1 == 0:
                        print("Epoch: ", epoch + 1, flush=True)#, " Loss: ", loss.numpy(), flush=True)
                    
                    get_train_acc_tf((images, labels))
                    train_accuracy = train_acc_metric.result()
                    train_acc_list.append(train_accuracy.numpy())
                    print("train accuracy ", train_accuracy.numpy(), flush=True)
                    get_test_acc_tf((images_test, labels_test))
                    test_accuracy = test_acc_metric.result()
                    test_acc_list.append(test_accuracy.numpy())
                    print("test accuracy ", test_accuracy.numpy(), flush=True)

                    flat_original_grads = self.undo_gradient_normalize(self.flatten_gradients(o_grads), num_samples_orig)


                    print("original magnitude ", np.linalg.norm(flat_original_grads), flush=True)
                    flat_alt_grads = self.undo_gradient_normalize(self.flatten_gradients(a_grads), num_samples_alt)


                    print("alt magnitude ", np.linalg.norm(flat_alt_grads), flush=True)
                    current_sensitivity = self.compute_gradient_norm(flat_original_grads, flat_alt_grads)
                    print("current sensitivity ", current_sensitivity, flush=True)
                    print("l2_norm ", self.l2_norm_clip, flush=True)
                    if local:
                        print('local', flush=True)
                        current_sensitivity = self.compute_gradient_norm(flat_original_grads, flat_alt_grads)
                        priv_opt.set_sensitivity(current_sensitivity)
                        if worst_case:
                            tails_flat = self.worst_case_noise(flat_gradients=flat_original_grads,
                                                            flat_alt_grad=flat_alt_grads,
                                                            local_sensitivity=current_sensitivity)
                            tails_and_vars = self.reshape_tails(tails_flat=tails_flat,
                                                                gradients=o_grads)
                            tails = [tail for tail, _ in tails_and_vars]
                            priv_opt.set_tail(tails)
                    elif bounded:
                        priv_opt.set_sensitivity(2*self.l2_norm_clip)
                    priv_grad_val = distributed_grad((images, labels))
                    flat_private_grads = self.undo_gradient_normalize(self.flatten_gradients(priv_grad_val), num_samples_orig)
                    print('sensitivity adding noise ', priv_opt.get_sensitivity(), flush=True)
                    print("magnitude nosie added ", self.compute_gradient_norm(flat_original_grads, flat_private_grads), flush=True)
                    original_grad_list.append(flat_original_grads)
                    alt_grad_list.append(flat_alt_grads)
                    private_grad_list.append(flat_private_grads)
                test_acc_metric.reset_states()
                train_acc_metric.reset_states()                    
        return original_grad_list, alt_grad_list, private_grad_list, train_acc_list, test_acc_list

    
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
        test_acc_metric = tf1.keras.metrics.CategoricalAccuracy()
        test_logits = self.keras_model(test_x_placeholder)
        calc_test_acc = test_acc_metric.update_state(y_test, test_logits)
        test_acc = test_acc_metric.result()

        accuracy = tf1.keras.metrics.CategoricalAccuracy()
        update_acc = accuracy.update_state(labels, probs)
        get_acc = accuracy.result()

        sess = tf1.keras.backend.get_session()
        Model_Purch.init_uninitialized_vars(sess)

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
            Model_Purch.init_uninitialized_vars(sess)

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


if __name__ == "__main__":
    import tensorflow.keras.datasets.mnist as mnist
    from dpa.attacker import GaussianAttacker
    from dpa.evaluation import RepeatedAttack
    #uncomment to use non-eager training
    #tf.compat.v1.disable_v2_behavior()

    adversary = GaussianAttacker()
    eval = RepeatedAttack()
    n_train=100
    bounded=False
    x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test = eval.get_train_test_data(bounded=bounded, white_picture=False, n_train=n_train)

    epochs = 3
    #composed_delta = 1 / n_train  # the delta you want in the end
    composed_delta = 0.01
    delta_i = composed_delta / epochs

    rho = 0.9
    local = True
    worst_case = False
    composed_eps = adversary.get_epsilon_for_confidence_bound(rho)

    if worst_case:
        # sequential composition due, only valid if worst case noise enforced due to loose bounds
        noise_multiplier = adversary.get_noise_multiplier_for_seq_eps(composed_eps, composed_delta,
                                                                      epochs)
        expected_success_rate = adversary.expected_seq_success_rate(composed_eps, composed_delta,
                                                                    epochs)
    else:
        #rdp world
        noise_multiplier = adversary.get_noise_multiplier_for_rdp_eps(composed_eps, composed_delta,
                                                                      epochs)
        expected_success_rate = adversary.expected_rdp_success_rate(noise_multiplier,
                                                                    epochs)
    print(noise_multiplier, flush=True)
    learning_rate = 0.005
    l2_norm_clip = 3
    print(f'Adversary confidence: {adversary.get_confidence_bound(composed_eps)}', flush=True)
    print(f'Adversary success rate: {expected_success_rate}', flush=True)
    print(f'main test', flush=True)

    final_confs = []

    for i in range(10):
        # fit the model
        m = Model_Purch()
        m.set_training_params(noise_multiplier=noise_multiplier,
                              learning_rate=learning_rate,
                              l2_norm_clip=l2_norm_clip,
                              epochs=epochs,
                              delta=composed_delta)

        results = m.train_eager(train_set=(x_train_orig, y_train_orig),
                          alternative_set=(x_train_alt, y_train_alt),
                          test_set = (x_test, y_test),
                          local=local,
                          worst_case=worst_case, bounded=bounded)

        original, alt, private = results
        # sensitivities
        if local:
            sensitivities = [np.linalg.norm(np.subtract(origin, alter))
                             for (origin, alter) in zip(original, alt)]
        elif bounded:
            sensitivities = 2*l2_norm_clip
        else:
            sensitivities = l2_norm_clip

        sigmas = np.multiply(noise_multiplier, sensitivities)
        biased, _ = adversary.infer(original, alt, private, sigmas)

        print("Belief iteration ", i, ": ", biased[0][-1], flush=True)
        final_confs.append(biased[0][-1])
        tf.keras.backend.clear_session()