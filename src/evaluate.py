# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule

# own modules
from data import load_data
from utils import (
    Accuracy,
    load_model,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"


def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """
    # Load the data
    _, _, test_loader = load_data(path=DATA_PATH, batch_size=32, n_patches=16)

    print(f"Data loaded")

    # Load the model
    model: RecursiveScriptModule = load_model(name)

    print(f"Model loaded")

    # Create the accuracy object
    accuracy = Accuracy()

    model.eval()
    # Iterate over the test_loader
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        accuracy.update(output, target)

    return accuracy.compute()


if __name__ == "__main__":
    print(f"accuracy: {main('model_1')}")
