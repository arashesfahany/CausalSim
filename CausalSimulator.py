import numpy as np


class CausalSim(object):
    trained: bool
    chosen_kappa = float

    def __init__(self, kappa_range):
        """
        Initiate a CausalSim object. The CausalSim object has to go through two phases before begin ready for inference:
        1) Training with a range of kappa
        2) Choosing a kappa

        Then, the object can be used single steps. The output of each step can be used as is or for constructing a more
        complex observation space, which is context-dependent. For instance, the construction can be aggregating the
        past
        :param kappa_range: A range of values for kappa to train with
        """
        pass

    def train_with_factual_data(self, obs_s: np.ndarray, act_s: np.ndarray, emit_s: np.ndarray, pol_s: np.ndarray):
        """
        Train the CausalSim object with factual data

        :param obs_s: Factual observations
        :param act_s: Factual actions
        :param emit_s: Factual emissions
        :param pol_s: Factual policies for each training points
        """
        self.trained = True
        pass

    def predict_step_with_latent(self, act_cf: int, obs_cf: np.ndarray, latent: float) -> np.ndarray:
        """
        This inference method uses an already computed latent variable to infer counterfactual observations
        :param act_cf: Counterfactual action
        :param obs_cf: Starting counterfactual observation
        :param latent: Latent variable, previously computed with CausalSim
        :return: Next counterfactual observation
        """
        assert self.trained
        assert self.chosen_kappa is not None
        pass

    def predict_step(self, act_cf: int, obs_cf: np.ndarray, emit_real: np.ndarray, act_real: int) -> np.ndarray:
        """
        This inference method computes the latent variable with factual data and infers counterfactual observations
        :param act_cf: Counterfactual action
        :param obs_cf: Starting counterfactual observation
        :param emit_real: Factual emission data for this step
        :param act_real: Factual action for this step
        :return: Next counterfactual observation
        """
        assert self.trained
        assert self.chosen_kappa is not None
        pass

    def _choose_kappa(self, cf_obs_s: np.ndarray, scoring_func):
        """
        Use the training data to choose kappa. The procedure uses out-of-distribution predictions that can be answered
        with the training data to optimize kappa.

        :param cf_obs_s: The out-of-distribution predictions sourced on training data and predicted using training
        policies, the first dimension corresponds to the source data policy and the second dimension corresponds to the
        target inferring policy.
        :param scoring_func: A scoring function used for evaluating CausalSim predictions.
        """
        assert self.trained
        self.chosen_kappa = 1
        pass

