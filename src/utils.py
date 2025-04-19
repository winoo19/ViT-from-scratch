# deep learning libraries
import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional


class StepLR(torch.optim.lr_scheduler.LRScheduler):
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """

    optimizer: torch.optim.Optimizer
    step_size: int
    gamma: float
    last_epoch: int
    counters: int

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float = 0.1,
    ) -> None:
        """
        This method is the constructor of StepLR class.

        Args:
            optimizer: optimizer.
            step_size: size of the step.
            gamma: factor to change the lr. Defaults to 0.1.
        """

        # TODO
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = -1
        self.counters = 0

    def step(self, epoch: Optional[int] = None) -> None:
        """
        This function is the step of the scheduler.

        Args:
            epoch: ignore this argument. Defaults to None.
        """

        # TODO
        self.last_epoch += 1

        if (self.last_epoch + 1) % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.gamma

            self.counters += 1


def get_dropout_random_indexes(shape: torch.Size, p: float) -> torch.Tensor:
    """
    This function get the indexes to put elements at zero for the
    dropout layer. It ensures the elements are selected following the
    same implementation than the pytorch layer.

    Args:
        shape: shape of the inputs to put it at zero. Dimensions: [*].
        p: probability of the dropout.

    Returns:
        indexes to put elements at zero in dropout layer.
            Dimensions: shape.
    """

    # get inputs indexes
    inputs: torch.Tensor = torch.ones(shape)

    # get indexes
    indexes: torch.Tensor = F.dropout(inputs, p)
    indexes = (indexes == 0).int()

    return indexes


class Accuracy:
    """
    This class is the accuracy object.

    Attr:
        correct: number of correct predictions.
        total: number of total examples to classify.
    """

    correct: int
    total: int

    def __init__(self) -> None:
        """
        This is the constructor of Accuracy class. It should
        initialize correct and total to zero.
        """

        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        This method update the value of correct and total counts.

        Args:
            logits: outputs of the model.
                Dimensions: [batch, number of classes]
            labels: labels of the examples. Dimensions: [batch].
        """

        # compute predictions
        predictions = logits.argmax(1).type_as(labels)

        # update counts
        self.correct += int(predictions.eq(labels).sum().item())
        self.total += labels.shape[0]

        return None

    def compute(self) -> float:
        """
        This method returns the accuracy value.

        Returns:
            accuracy value.
        """

        return self.correct / self.total

    def reset(self) -> None:
        """
        This method resets to zero the count of correct and total number of
        examples.
        """

        # init to zero the counts
        self.correct = 0
        self.total = 0

        return None


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
