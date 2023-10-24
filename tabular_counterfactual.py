"""
Tabular counterfactual is the implementation of the Counter factual search.

Original code from facebookresearch/mmd. It requires Numpy and typing to be
installed on the system.
"""

# Installation of the Counterfactual engine
# !git clone https://github.com/facebookresearch/mmd.git

import typing
import numpy as np
from mmd.counterfactuals.base_proxy import BasePerturbationProxy


class TabularPertubation(BasePerturbationProxy):
    """class to generate a basic engine for Tabular Data."""

    def __init__(self, model):
        self.initial_document = None  # used in document_to_perturbation_space
        self.model = model

    def classify(self, document: np.ndarray) -> typing.Tuple[bool, float]:
        """
        Create an classification of a given imput.

        Expects an numpy array as input.
        Returns a tuple out of label and score.
        The ML system is returning a Class label which is stored in label.
        With the prediciton a score is computed which can be interpreted as
        the certitude of the ML system in the predicted Label.
        """
        prediction = self.model.predict(document, verbose=0)
        score = np.absolute(prediction - 0.5) * 2
        label = np.round(prediction, 0)
        return_variable = (label, score)
        return return_variable

    def document_to_perturbation_space(self, document: np.ndarray) -> typing.List:
        """
        Create an pertubation spaces.

        Which in this case converts the numpy array into a list.
        Expects an numpy array as input.
        Returns a List.
        """
        self.initial_document = document
        perturbation_space = document.tolist()
        return perturbation_space


class TabularSimplePertubation(TabularPertubation):
    """
    Class to generate a simple pertubation on the ML system.

    Simple in this case means that the perturbed sequence ist set 0.
    """

    def perturb_positions(
        self, perturbation_space: typing.List, positions: typing.List[int]
    ) -> np.ndarray:
        """
        Perturbin the input to 0.

        Function which generates an output np.array.
        The output array contains 0 in the given positions in
        the according pertubation space.
        """
        perturbed_sequence = []
        for i,_ in enumerate(perturbation_space):
            if i not in positions:
                perturbed_sequence.append(perturbation_space[i])
            else:
                perturbed_sequence.append(0)
        return np.array(perturbed_sequence)


class TabularMeanPertubation(TabularPertubation):
    """
    Class to generate a mean pertubation on the ML system.

    Mean in this case means that the perturbed sequence ist set to
    the mean of the specific variable to be perturbed.
    """

    def __init__(self, model, train_dataset: np.ndarray):
        super().__init__(model)
        self.train_dataset = train_dataset
        self.length_of_trainingdata = self.train_dataset.shape[1]
        # creating an empty numpy array to be filled
        self.dst_means = np.zeros(self.length_of_trainingdata)
        # filling the empty array with the mean values
        # of the columns of the test dataset
        for ind in range(self.length_of_trainingdata):
            self.dst_means[ind] = self.train_dataset[:, ind].mean()

    def perturb_positions(
        self, perturbation_space: typing.List, positions: typing.List[int]
    ) -> np.ndarray:
        """
        Perturbin the input to mean.

        Function which generates an output np.array.
        The output array contains means of the colums of the test dataset
        in the given positions in the according pertubation space.
        """
        perturbed_sequence = []
        for i,_ in enumerate(perturbation_space):
            if i not in positions:
                perturbed_sequence.append(perturbation_space[i])
            else:
                perturbed_sequence.append(self.dst_means[i])
        return np.array(perturbed_sequence)


class TabularMedianPertubation(TabularPertubation):
    """
    Class to generate a median pertubation on the ML system.

    Median in this case means that the perturbed sequence ist set to
    the median of the specific variable to be perturbed.
    """

    def __init__(self, model, train_dataset: np.ndarray):
        super().__init__(model)
        self.train_dataset = train_dataset
        self.length_of_trainingdata = self.train_dataset.shape[1]
        # creating an empty numpy array to be filled
        self.dst_medians = np.zeros(self.length_of_trainingdata)
        # filling the empty array with the median values
        # of the columns of the test dataset
        for ind in range(self.length_of_trainingdata):
            self.dst_medians[ind] = np.median(self.train_dataset[:, ind])

    def perturb_positions(
        self, perturbation_space: typing.List, positions: typing.List[int]
    ) -> np.ndarray:
        """
        Perturbin the input to median.

        Function which generates an output np.array.
        The output array contains median of the colums of the test dataset
        in the given positions in the according pertubation space.
        """
        perturbed_sequence = []
        for i,_ in enumerate(perturbation_space):
            if i not in positions:
                perturbed_sequence.append(perturbation_space[i])
            else:
                perturbed_sequence.append(self.dst_medians[i])
        return np.array(perturbed_sequence)


class TabularRandomUniformPertubation(TabularPertubation):
    """
    Class to generate a random pertubation on the ML system.

    Ramdom in this case means that the perturbed sequence ist set to a random
    state with uniform distribution of the specific variable to be perturbed.
    """

    def __init__(self, model, train_dataset: np.ndarray):
        super().__init__(model)
        self.train_dataset = train_dataset
        self.length_of_trainingdata = self.train_dataset.shape[1]
        # creating an empty numpy array to be filled
        self.uni_randoms = np.zeros(self.length_of_trainingdata)

    def perturb_positions(
        self, perturbation_space: typing.List, positions: typing.List[int]
    ) -> np.ndarray:
        """
        Perturbin the input to random uniform distribution.

        Function which generates an output np.array.
        The output array contains a random uniform distributed value
        of the max and min value of the colums of the test dataset,
        in the given positions in the according pertubation space.
        """
        # filling the empty array with random values
        for ind in range(self.length_of_trainingdata):
            low_ind = self.train_dataset[:, ind].min()
            high_ind = self.train_dataset[:, ind].max()
            self.uni_randoms[ind] = np.random.uniform(low=low_ind, high=high_ind)
        perturbed_sequence = []
        for i,_ in enumerate(perturbation_space):
            if i not in positions:
                perturbed_sequence.append(perturbation_space[i])
            else:
                perturbed_sequence.append(self.uni_randoms[i])
        return np.array(perturbed_sequence)


class TabularRandomNormalPertubation(TabularPertubation):
    """
    Class to generate a random pertubation on the ML system.

    Ramdom in this case means that the perturbed sequence
    is set to a random state with normal (Gaussian) distribution
    of the specific variable to be perturbed.
    """

    def __init__(self, model, train_dataset: np.ndarray):
        super().__init__(model)
        self.train_dataset = train_dataset
        self.length_of_trainingdata = self.train_dataset.shape[1]
        # creating an empty numpy array to be filled
        self.nor_randoms = np.zeros(self.length_of_trainingdata)

    def perturb_positions(
        self, perturbation_space: typing.List, positions: typing.List[int]
    ) -> np.ndarray:
        """
        Perturbin the input to random normal distribution.

        Function which generates an output np.array.
        The output array contains a random normal distributed value
        of the colums of the test dataset,
        in the given positions in the according pertubation space.
        """
        # filling the empty array with random values
        for ind in range(self.length_of_trainingdata):
            mean_ind = np.mean(self.train_dataset[:, ind].mean())
            std_ind = np.std(self.train_dataset[:, ind].max())
            self.nor_randoms[ind] = np.random.normal(loc=mean_ind, scale=std_ind)
        perturbed_sequence = []
        for i,_ in enumerate(perturbation_space):
            if i not in positions:
                perturbed_sequence.append(perturbation_space[i])
            else:
                perturbed_sequence.append(self.nor_randoms[i])
        return np.array(perturbed_sequence)
