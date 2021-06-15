from distutils.version import LooseVersion

import tensorflow as tf
import tensorflow_privacy.privacy.analysis.privacy_ledger as privacy_ledger
from tensorflow_privacy.privacy.dp_query.gaussian_query import GaussianSumQuery
from tensorflow_privacy.privacy.optimizers.dp_optimizer import make_optimizer_class


class LocalGaussianSumQuery(GaussianSumQuery):

    def __init__(self, l2_norm_clip, noise_multi, worst_case=False):
        self.worst_case = worst_case
        self.tail = None
        self.sensitivity = l2_norm_clip
        self.noise_multiplier = noise_multi
        self.norm_clip = l2_norm_clip
        stddev = self.noise_multiplier * l2_norm_clip
        super().__init__(l2_norm_clip, stddev)

    def set_tail(self, tail):
        self.tail = tail

    def set_sensitivity(self, sensitivity):
        # sensitivity = min(sensitivity, self.norm_clip)
        #print(sensitivity)
        self.sensitivity = sensitivity
        self._stddev = self.noise_multiplier * self.sensitivity

    def preprocess_record_impl(self, params, record):
        """Clips the l2 norm, returning the clipped record and the l2 norm.

        Args:
          params: The parameters for the sample.
          record: The record to be processed.

        Returns:
          A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
            the structure of preprocessed tensors, and l2_norm is the total l2 norm
            before clipping.
        """
        l2_norm_clip = self._l2_norm_clip#self.sensitivity
        record_as_list = tf.nest.flatten(record)
        clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
        return tf.nest.pack_sequence_as(record, clipped_as_list), norm

    def get_noised_result(self, sample_state, global_state):
        """See base class."""
        if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
            def add_random_noise(v):
                return v + tf.random_normal(tf.shape(v), stddev=self._stddev)
        else:
            random_normal = tf.compat.v1.random_normal_initializer(
                stddev=self._stddev)
            def add_random_noise(v):
                return v + random_normal(tf.shape(v))

        def add_tailbound_noise(v, tail):
            if self.noise_multiplier == 0:
                return v + tf.zeros_like(tail)
            else:
                return v + tail

        if self._ledger:
            dependencies = [
                self._ledger.record_sum_query(
                    global_state.l2_norm_clip, global_state.stddev)
            ]
        else:
            dependencies = []

        with tf.control_dependencies(dependencies):
            if self.worst_case:
                noised_grads = tf.nest.map_structure(add_tailbound_noise, sample_state, self.tail)
                return noised_grads, global_state
            else:
                return tf.nest.map_structure(add_random_noise, sample_state), global_state


def make_local_gaussian_optimizer_class(cls):
  """Constructs a DP optimizer with Gaussian averaging of updates."""

  class LocalDPGaussianOptimizerClass(make_optimizer_class(cls)):
    """DP subclass of given class cls using Gaussian averaging."""

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        ledger=None,
        unroll_microbatches=False,
        worst_case=False,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs):
        dp_sum_query = LocalGaussianSumQuery(
            l2_norm_clip, noise_multiplier, worst_case=worst_case)

        if ledger:
            dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query, ledger=ledger)

        super(LocalDPGaussianOptimizerClass, self).__init__(
            dp_sum_query,
            num_microbatches,
            unroll_microbatches,
            *args,
            **kwargs)

    @property
    def ledger(self):
        return self._dp_sum_query.ledger

    def set_tail(self, tail):
        self._dp_sum_query.set_tail(tail)

    def set_sensitivity(self, sensitivity):
        self._dp_sum_query.set_sensitivity(sensitivity)

    def get_sensitivity(self):
        return self._dp_sum_query.sensitivity

  return LocalDPGaussianOptimizerClass


AdamOptimizer = tf.compat.v1.train.AdamOptimizer
GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer

LocalDPAdamGaussianOptimizer = make_local_gaussian_optimizer_class(AdamOptimizer)
LocalDPGradientDescentGaussianOptimizer = make_local_gaussian_optimizer_class(
    GradientDescentOptimizer)
