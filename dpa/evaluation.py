from dpa.attacker import GaussianAttacker
from dpa.purch_query import Model_Purch
from dpa.adult_query import Model_Adult
from dpa.query import Model

import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
from random import shuffle
from os import path
from tensorflow.keras.datasets.mnist import load_data
#from skimage.metrics import structural_similarity as ssim


class RepeatedAttack:

    def synthetic_attack(self, dimensions, epochs, upperbound, composed_delta, worst, sequential, iterations):
        adversary = GaussianAttacker()
        final_confs = []
        for i in range(iterations):
            a = (np.random.randint(10, size=(2,dimensions)))
            composed_epsilon = adversary.get_epsilon_for_confidence_bound(upperbound)
            compare = []
            for i in range(0,2):
                compare.append(np.repeat([a[i]], epochs, axis=0))
            sensitivity_synthetic = np.linalg.norm(a[0]-a[1])
            if worst:
                epsilon = composed_epsilon/epochs
                sigma = sensitivity_synthetic * adversary.get_noise_multiplier_for_seq_eps(composed_epsilon, composed_delta, epochs)
                magnitude = epsilon*(sigma)**2/sensitivity_synthetic- sensitivity_synthetic/2
                direction = (a[0]-a[1])/np.linalg.norm(a[0]-a[1])
                tails = (direction*magnitude)
                private = a[0]+tails
                compare.append(np.repeat([private], epochs, axis=0))
            else:
                if sequential:
                    sigma_part = adversary.get_noise_multiplier_for_seq_eps(composed_epsilon, composed_delta, epochs)
                else:
                    sigma_part = adversary.get_noise_multiplier_for_rdp_eps(composed_epsilon, composed_delta, epochs)
                sigma = sigma_part * sensitivity_synthetic
                comp = []
                for z in range(0, epochs):
                    private = []
                    for j in range(0,len(a[0])):
                        private.append(a[0][j] +np.random.normal(0, sigma))
                    comp.append(private)
                compare.append(comp)
            beliefs_synthetic, beliefs_synthetic_noprior = adversary.infer(compare[0], compare[1], compare[2], sigma)
            final_confs.append(beliefs_synthetic[0][-1])
        return final_confs

    def load_purch_train_test_data(self, orig, alt, filepath):
        orig = np.load(path.join(filepath, f"{orig}.npz"), allow_pickle=True)["lookup"]
        alt = np.load(path.join(filepath, f"{alt}.npz"), allow_pickle=True)["lookup"]

        output = [index for index, elem in enumerate(orig) if (elem != alt[index]).any()][0]
        print(output)

        all_data = np.load('../../../data/purch/shokri_purchases_100_classes.npz')
        X = all_data['x'][:80000]
        X = X.reshape((len(X), 600))
        Y = tf.keras.utils.to_categorical(all_data['y'][:80000])

        index = -1
        for i in range(len(X)):
            if (X[i]==alt[output]).all():
                index = i

        x_train_orig = X[:len(orig)]
        x_train_alt = np.copy(X[:len(orig)])
        x_train_alt[output] = X[index]
        y_train_orig = Y[:len(orig)]
        y_train_alt = np.copy(Y[:len(orig)])
        y_train_alt[output] = Y[index]
        x_test = X[60000:]
        y_test = Y[60000:]
        if index > 60000:
            x_test[index-60000] = X[59999]
            y_test[index-60000] = Y[59999]


        return x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test

    def load_adult_train_test_data(self, orig, alt, filepath):
        orig = np.load(path.join(filepath, f"{orig}.npz"), allow_pickle=True)["lookup"]
        alt = np.load(path.join(filepath, f"{alt}.npz"), allow_pickle=True)["lookup"]

        output = [index for index, elem in enumerate(orig) if (elem != alt[index]).any()][0]
        #print(output)

        all_data = np.load('../../../data/adult/adult_wo_na.npz')
        X = all_data['x']
        X = X.reshape((len(X), 104))
        Y = np.load('../../../data/adult/adult_wo_na_labels.npz')['y']

        index = -1
        for i in range(len(X)):
            if (X[i]==alt[output]).all():
                index = i
        
        # Train size from original paper 30162
        # Test size from original paper 15060
        x_train_orig = X[:len(orig)]
        x_train_alt = np.copy(X[:len(orig)])
        x_train_alt[output] = X[index]
        y_train_orig = Y[:len(orig)]
        y_train_alt = np.copy(Y[:len(orig)])
        y_train_alt[output] = Y[index]
        x_test = X[30162:]
        y_test = Y[30162:]


        return x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test

    def load_train_test_data(self, orig, alt, filepath):
        orig = np.load(path.join(filepath, f"{orig}.npz"), allow_pickle=True)["lookup"]
        alt = np.load(path.join(filepath, f"{alt}.npz"), allow_pickle=True)["lookup"]

        output = [index for index, elem in enumerate(orig) if (elem != alt[index]).any()][0]
        print(output)

        (x_train, y_train), (x_test, y_test) = load_data()
        X = np.concatenate((x_train, x_test))
        X = X.reshape((len(X), 28 * 28))
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        x_train = (x_train[...,None] / 255)
        x_test = (x_test[...,None] / 255)

        index = -1
        for i in range(len(X)):
            if (X[i]==alt[output]).all():
                index = i
        

        x_train_orig = x_train[:len(orig)]
        x_train_alt = np.copy(x_train[:len(orig)])
        x_train_alt[output] = x_train[index]
        y_train_orig = y_train[:len(orig)]
        y_train_alt = np.copy(y_train[:len(orig)])
        y_train_alt[output] = y_train[index]
        if index > 60000:
            x_test[index-60000] = x_train[-1]
            y_test[index-60000] = y_train[-1]


        return x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test

    def load_unbounded_adult_train_test_data(self, orig, alt, filepath):
        orig = np.load(path.join(filepath, f"{orig}.npz"), allow_pickle=True)["lookup"]
        alt = np.load(path.join(filepath, f"{alt}.npz"), allow_pickle=True)["lookup"]
        all_data = np.load('../../../data/adult/adult_wo_na.npz')
        X = all_data['x']
        Y = np.load('../../../data/adult/adult_wo_na_labels.npz')['y']

        index = -1
        for i in range(len(X)):
            if not (X[i]==alt[i]).all():
                index = i
                break

        x_train_orig = X[:len(orig)]
        x_train_alt = np.copy(X[:len(orig)])
        x_train_alt = np.delete(x_train_alt, index,0)

        y_train_orig = Y[:len(orig)]
        y_train_alt = np.copy(Y[:len(orig)])
        y_train_alt = np.delete(y_train_alt, index,0)

        x_test = X[30162:]
        y_test = Y[30162:]

        return x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test

    def load_unbounded_purch_train_test_data(self, orig, alt, filepath):
        orig = np.load(path.join(filepath, f"{orig}.npz"), allow_pickle=True)["lookup"]
        alt = np.load(path.join(filepath, f"{alt}.npz"), allow_pickle=True)["lookup"]
        all_data = np.load('../../../data/purch/shokri_purchases_100_classes.npz')
        X = all_data['x'][:10000]
        Y = tf.keras.utils.to_categorical(all_data['y'][:10000])

        index = -1
        for i in range(len(X)):
            if not (X[i]==alt[i]).all():
                index = i
                break
        
        #if index > 8000:
        #    raise Exception('must change test data, adjust implementation')


        x_train_orig = X[:len(orig)]
        x_train_alt = np.copy(X[:len(orig)])
        x_train_alt = np.delete(x_train_alt, index,0)

        y_train_orig = Y[:len(orig)]
        y_train_alt = np.copy(Y[:len(orig)])
        y_train_alt = np.delete(y_train_alt, index,0)

        x_test = X[8000:]
        y_test = Y[8000:]

        return x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test

    def load_unbounded_train_test_data(self, orig, alt, filepath):
        orig = np.load(path.join(filepath, f"{orig}.npz"), allow_pickle=True)["lookup"]
        alt = np.load(path.join(filepath, f"{alt}.npz"), allow_pickle=True)["lookup"]

        (x_train, y_train), (x_test, y_test) = load_data()
        X = np.concatenate((x_train, x_test))
        X = X.reshape((len(X), 28 * 28))
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        x_train = (x_train[...,None] / 255)
        x_test = (x_test[...,None] / 255)

        index = -1
        for i in range(len(orig)):
            if not (X[i]==alt[i]).all():
                index = i
                break

        x_train_orig = x_train[:len(orig)]
        x_train_alt = np.copy(x_train[:len(orig)])
        x_train_alt = np.delete(x_train_alt, index,0)

        y_train_orig = y_train[:len(orig)]
        y_train_alt = np.copy(y_train[:len(orig)])
        y_train_alt = np.delete(y_train_alt, index,0)

        return x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test

    def nn_attack_graph(self, epochs, composed_delta, rho, worst_case, local, sequential, iterations, learning_rate, l2_norm_clip, bounded, MNIST, ADULT, PURCH):
        adversary = GaussianAttacker()

        if bounded:
            print("Running bounded")
            if MNIST:
                n_train = 100
                x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test = self.load_train_test_data(orig="original_data_set_ssim_size_100_place_1", alt="alt_data_set_ssim_size_100_place_1", filepath="../../../data/mnist/ssim/")
            elif PURCH:
                n_train = 1000
                x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test = self.load_purch_train_test_data(orig="original_data_set_ham_size_100_place_1", alt="alt_data_set_ham_size_1000_place_1", filepath="../../../data/purch/pca/")
            elif ADULT:
                n_train = 1000
                x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test = self.load_adult_train_test_data(orig="original_data_set_adult_wo_na_size_1000_place_1", alt="alt_data_set_adult_wo_na_size_1000_place_1", filepath="../../../data/adult/")
        else:
            print("Running unbounded")
            if MNIST:
                n_train = 100
                x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test = self.load_unbounded_train_test_data(orig="unbounded_original_data_set_ssim_size_100", alt="unbounded_alt_data_set_ssim_size_100", filepath="../../../data/mnist/ssim/")
            elif PURCH:
                n_train = 1000
                x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test = self.load_unbounded_purch_train_test_data(orig="unbounded_original_data_set_ham_size_1000", alt="unbounded_alt_data_set_ham_size_1000", filepath="../../../data/purch/hamming/")
            elif ADULT:
                n_train = 1000
                x_train_orig, x_train_alt, y_train_orig, y_train_alt, x_test, y_test = self.load_unbounded_adult_train_test_data(orig="unbounded_original_adult_data_set_manh_size_1000", alt="unbounded_alt_adult_data_set_manh_size_1000", filepath="../../../data/adult/")

        
        composed_eps = adversary.get_epsilon_for_confidence_bound(rho)

        if worst_case:
            # sequential composition due, only valid if worst case noise enforced due to loose bounds
            noise_multiplier = adversary.get_noise_multiplier_for_seq_eps(composed_eps, composed_delta, epochs)
            expected_success_rate = adversary.expected_seq_success_rate(composed_eps, composed_delta, epochs)
        else:
            if sequential:
                noise_multiplier = adversary.get_noise_multiplier_for_seq_eps(composed_eps, composed_delta, epochs)
                expected_success_rate = adversary.expected_seq_success_rate(composed_eps, composed_delta, epochs)
            else:
                #rdp world
                noise_multiplier = adversary.get_noise_multiplier_for_rdp_eps(composed_eps, composed_delta, epochs)
                expected_success_rate = adversary.expected_rdp_success_rate(noise_multiplier,epochs)

        run_summary = {}
        
        print(f'Adversary confidence: {adversary.get_confidence_bound(composed_eps)}')
        print(f'Adversary success rate: {expected_success_rate}')
        
        for i in range(iterations):
            # fit the model
            if MNIST:
                print("Running MNIST")
                m = Model()
            elif PURCH:
                print("Running PURCHASE-100")
                m = Model_Purch()
            elif ADULT:
                print("Running Adult")
                m = Model_Adult()
            m.set_training_params(noise_multiplier=noise_multiplier,
                                learning_rate=learning_rate,
                                l2_norm_clip=l2_norm_clip,
                                epochs=epochs,
                                delta=composed_delta)
                                
            results = m.train(train_set=(x_train_orig, y_train_orig),
                            alternative_set=(x_train_alt, y_train_alt), 
                            test_set=(x_test, y_test),
                            local=local, 
                            worst_case=worst_case,
                            bounded=bounded)
            
            original, alt, private, train_acc, test_acc = results
            sensitivities = [np.linalg.norm(np.subtract(origin, alter))
                            for (origin, alter) in zip(original, alt)]
            #print("sensitivities ", sensitivities)
            # sensitivities
            if local:
                sensitivities = [np.linalg.norm(np.subtract(origin, alter))
                            for (origin, alter) in zip(original, alt)]
            elif bounded:
                sensitivities = 2*l2_norm_clip
            else:
                sensitivities = l2_norm_clip
            print("sensitivities ", sensitivities)


            
            sigmas = np.multiply(noise_multiplier, sensitivities)
            biased, unbiased = adversary.infer(original, alt, private, sigmas)

            run_summary[f"run_{i}"] = {"biased_beliefs": biased, "unbiased_beliefs": unbiased, "sensitivites": sensitivities, "train_acc":train_acc, "test_acc":test_acc}
            #m.save('./experiments/local_sensitivity_mnist/model.h5')
            print("Belief iteration ", i, ": ", biased[0][-1])
            tf.keras.backend.clear_session()
        return run_summary
