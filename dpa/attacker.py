from abc import abstractmethod, ABC

import numpy as np
from scipy.optimize import bisect
from scipy.stats.distributions import norm
from tensorflow_privacy.privacy.analysis.rdp_accountant \
    import compute_rdp, get_privacy_spent


class Attacker(ABC):
    @abstractmethod
    def expected_success_rate(self, epsilon, delta):
        """
        Returns the ratio of how many guesses will be correct
        :param epsilon:
        :param delta:
        :return:
        """
        pass

    @abstractmethod
    def infer(self, results_1, results_2, private_results, noise_params, prior_1=0.5):
        """
        Runs the attacker for a series of query results
        :param results_1: the sequence of original results computed on data set 1
        :param results_2: the sequence of original results computed on data set 2
        :param private_results: the sequence of results obtained from a mechanism that run on data
                                set 1 or 2
        :param noise_params: parameters of the mechanism, if provided as scalar the same
                             configuration is used for every result. Otherwise, a list can be
                             provided, that matches the length of the result sequences.
        :param prior_1: initial prior belief towards data set 1, prior belief for data set 2 is
                        computed as 1 - prior_1
        :return:
        """
        pass

    def get_confidence_bound(self, epsilon):
        """
        Returns the maximum confidence the attacker will have in any event
        :param epsilon:
        :return:
        """
        return 1. / (1 + np.power(np.e, -epsilon))

    def get_epsilon_for_confidence_bound(self, rho):
        """
        Returns the maximum rdp composed epsilon
        :param rho:
        :return:
        """
        return np.log(rho/(1-rho))

    def get_noise_multiplier_for_seq_eps(self, composed_eps, composed_delta, epochs):
        """
        Returns the noise multiplier that satisfies a rdp composed epsilon and delta
        :param rdp_eps:
        :param composed_delta:
        :param epochs:
        :return:
        """
        epsilon_i = composed_eps/epochs
        delta_i = composed_delta/epochs
        return 1/epsilon_i*np.sqrt(2*np.log(1.25/delta_i))

    def compute_rdp_eps(self, noise_multiplier, epochs, delta):
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        sampling_probability = 1.0
        rdp = compute_rdp(q=sampling_probability,
                          noise_multiplier=noise_multiplier,
                          steps=epochs * 1.,
                          orders=orders)

        rdp_eps = get_privacy_spent(orders, rdp, target_delta=delta)[0]
        return rdp_eps

    def get_noise_multiplier_for_rdp_eps(self, composed_eps, composed_delta, epochs):
        """
        Returns the noise multiplier that satisfies a rdp composed epsilon and delta
        :param composed_eps:
        :param composed_delta:
        :param epochs:
        :return:
        """
        # perform binary search to find possible solution
        # shift required since interval has to be neg, pos
        shift = 1024.
        # upper and lower bound for noise_multiplier
        a = 0.01
        b = shift*2

        def f(x):
            return self.compute_rdp_eps(x + shift, epochs, composed_delta) - composed_eps

        print(f)
        noise_multiplier = bisect(f, a - shift, b - shift) + shift
        return noise_multiplier

    def get_expected_confidence_bound(self, epsilon, delta):

        """
        Returns the expected maximum confidence the attacker will have in any event
        :param epsilon:
        :param delta:
        :return:
        """
        p_c = self.get_confidence_bound(epsilon)
        return p_c + delta * (1 - p_c)


class GaussianAttacker(Attacker):
    """
    Attacker definition targeting the Gaussian mechanism
    """
    def expected_success_rate(self, epsilon, delta):
        pass

    def expected_seq_success_rate(self, epsilon, delta, k):
        """
        Computes the expected success rate single and multiple dimensions (not composition)
        :param epsilon:
        :param delta:
        :return:
        """
        return 1 - norm.cdf(-epsilon * np.sqrt(k)/ (2 * np.sqrt(2 * np.log(1.25 / delta))))

    def expected_rdp_success_rate(self, noise_multiplier, k):
        """
        Computes the expected success rate single and multiple dimensions (not composition)
        :param epsilon:
        :param delta:
        :return:
        """
        print("changed")
        return norm.cdf(np.sqrt(k)/(2*noise_multiplier))

    def infer(self, results_1, results_2, private_results, noise_params, prior_1=0.5):
        """
        Runs the attacker for a series of query results
        :param results_1: the sequence of original results computed on data set 1
        :param results_2: the sequence of original results computed on data set 2
        :param private_results: the sequence of results obtained from a gaussian mechanism
                                that run on data set 1 or 2
        :param noise_params: standard deviation(s) used in the gaussian mechanism. If provided as
                             scalar the same configuration is used for every result.
                             Otherwise, a list can be provided, that matches the length of the
                             result sequences.
        :param prior_1: initial prior belief towards data set 1, prior belief for data set 2 is
                        computed as 1 - prior_1
        :return:
        """

        if len(results_1) != len(results_2) != len(private_results):
            raise ValueError("The length of all results arrays must match")

        try:
            _ = iter(noise_params)
            if len(noise_params) != len(results_1):
                raise ValueError("The length of noise parameters must match "
                                 "the length of result arrays")
        except TypeError:
            noise_params = [noise_params for _ in range(len(results_1))]

        prior_2 = 1 - prior_1

        # guesses and beliefs based on unbiased priors of 0.5
        unbiased_guesses, unbiased_beliefs_1, unbiased_beliefs_2 = [], [0.5], [0.5]

        # guesses and beliefs based on previously seen query results, adaptive priors
        biased_guesses, biased_beliefs_1, biased_beliefs_2 = [], [0.5], [0.5]

        # start guessing using bay'sian beliefs
        # for numerical stability, use the fact that:
        # prior1 * pdf1 / (prior1 * pdf1 + prior2 * pdf2)
        # = prior1 / (prior1 + prior2 * pdf2 / pdf1)
        for (result_1, result_2, private_result, std) in zip(results_1, results_2,
                                                             private_results, noise_params):
            # compute relevant parts of gaussian pdf
            diff_1 = private_result - result_1
            diff_2 = private_result - result_2
            exponent_1 = np.dot(diff_1, diff_1) / (2 * (std ** 2))
            exponent_2 = np.dot(diff_2, diff_2) / (2 * (std ** 2))

            # dividing Gaussian pdfs results in subtraction of exponents
            pdf_1_div_2 = np.exp(exponent_2 - exponent_1)
            pdf_2_div_1 = np.exp(exponent_1 - exponent_2)
            
            biased_belief_1 = prior_1 / (prior_1 + prior_2 * pdf_2_div_1)
            biased_belief_2 = prior_2 / (prior_1 * pdf_1_div_2 + prior_2)

            prior_1 = biased_belief_1
            prior_2 = biased_belief_2

            biased_beliefs_1.append(biased_belief_1)
            biased_beliefs_2.append(biased_belief_2)

            # consider prior1 == prior2 for unbiased beliefs
            unbiased_belief_1 = 1 / (1 + pdf_2_div_1)
            unbiased_belief_2 = 1 / (pdf_1_div_2 + 1)
            unbiased_beliefs_1.append(unbiased_belief_1)
            unbiased_beliefs_2.append(unbiased_belief_2)

            biased_guesses.append(np.argmax([biased_belief_1, biased_belief_2]))
            unbiased_guesses.append(np.argmax([unbiased_belief_1, unbiased_belief_2]))

        biased_results = biased_beliefs_1, biased_beliefs_2, biased_guesses
        unbiased_results = unbiased_beliefs_1, unbiased_beliefs_2, unbiased_guesses

        return biased_results, unbiased_results

