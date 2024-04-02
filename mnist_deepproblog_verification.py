import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_fact_accuracy
from deepproblog.examples.minimal.data import AdditionDataset, MNISTImages
from deepproblog.examples.minimal.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from more_itertools import unzip
from pgd import pgd
from torch import float64
from torch import round as round_tensor
from torch.utils.data import DataLoader as TorchDataLoader

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

BATCH_SIZE = 32
PRINT = False
NUM_DIGIT_CLASSES = 10

RESULTS_COLUMNS = [
    "mnist_id",
    "classification_0_lb",
    "classification_0_ub",
    "classification_1_lb",
    "classification_1_ub",
    "classification_2_lb",
    "classification_2_ub",
    "classification_3_lb",
    "classification_3_ub",
    "classification_4_lb",
    "classification_4_ub",
    "classification_5_lb",
    "classification_5_ub",
    "classification_6_lb",
    "classification_6_ub",
    "classification_7_lb",
    "classification_7_ub",
    "classification_8_lb",
    "classification_8_ub",
    "classification_9_lb",
    "classification_9_ub",
    "classification_prediction_idx",  # Note: ouputs of NN are defined as 'predictions'
    "classification_target_idx",  # Note: labels are defined as 'targets'
    "classification_correct",
    "classification_correct_safe_pgd_attack_success",
    "classification_safe",
]
MODEL_PATH = Path(__file__).parent.resolve() / "checkpoints"
RESULTS_PATH = Path(__file__).parent.resolve() / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Simple MNIST based program to add 2 digits
PROGRAM_STRING = """ 
nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).
addition(X,Y,Z) :- digit(X,X2), digit(Y,Y2), Z is X2+Y2.
"""


def approx_lte(x, y, atol=1e-5):
    """Approximate less than or equal to a tensor with absolute tolerence

    Args:
        x (Tensor)
        y (Tensor)
        atol (_type_, optional): absolute tolerence. Defaults to 1e-5.

    Returns:
        bool: True / False
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=float64)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=float64)

    return (x <= y).all() or (torch.isclose(x, y, atol=atol)).all()


def approx_gte(x, y, atol=1e-5):
    """Approximate greater than or equal to a tensor with absolute tolerence

    Args:
        x (Tensor)
        y (Tensor)
        atol (_type_, optional): absolute tolerence. Defaults to 1e-5.

    Returns:
        bool: True / False
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=float64)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=float64)

    return (x >= y).all() or (torch.isclose(x, y, atol=atol)).all()


def bound_softmax(h_L, h_U, use_float64=False):
    """Given lower and upper input bounds into a softmax, calculate their concrete
    output bounds."""

    if use_float64:
        h_L = h_L.to(float64)
        h_U = h_U.to(float64)

    shift = h_U.max(dim=1, keepdim=True).values
    exp_L, exp_U = torch.exp(h_L - shift), torch.exp(h_U - shift)
    lower = exp_L / (
        torch.sum(exp_U, dim=1, keepdim=True) - exp_U + exp_L
    )  # TODO EdS: Check removed epsilon
    upper = exp_U / (torch.sum(exp_L, dim=1, keepdim=True) - exp_L + exp_U)

    return lower, upper


def get_mnist_deepproblog_model(network_module: torch.nn.Module) -> Model:
    """Read a trained MNIST model from a file if present, otherwise train the model and return a DeepProblog Model

    Args:
        network_module (torch.nn.Module): Torch Network module, the MNIST Net in this case

    Returns:
        Model: DeepProblog Model
    """
    net = Network(network_module, "mnist_net", batching=True)
    net.optimizer = torch.optim.Adam(network_module.parameters(), lr=1e-3)

    model = Model(PROGRAM_STRING, [net], load=False)
    model.set_engine(ExactEngine(model))
    model.add_tensor_source("train", MNISTImages("train"))
    model.add_tensor_source("test", MNISTImages("test"))

    dataset = AdditionDataset("train")

    # If there is no checkpoint train the model
    if not os.path.exists(MODEL_PATH):
        loader = DataLoader(dataset, 2, True)
        train_model(model, loader, 1, log_iter=100, profile=0)
        model.save_state(MODEL_PATH / "trained_model.pth")

    model.load_state(MODEL_PATH / "trained_model.pth")
    return model


def calculate_bounds(
    model: torch.nn.Module,
    dataloader,
    epsilon: float,
    method: str = "IBP",
    round_floats=False,
):
    """Calculate bounds for the provided model and the test dataset.

    In this scenario, the model is a simple MNIST digit classifier

    Args:
        model (torch.nn.Module): NN Module to be verified
        dataloader (_type_): Dataloader for the test dataset
        epsilon (float): perturbation measure
        method (str, optional): Method of approximating the bounds. Defaults to "IBP".
        round_floats (bool, optional): If the floats need to be rounded. Defaults to False.

    Raises:
        Exception: NotImplementedError, in case the method is not implemented by autoLirpa
    """
    print(f"Performing verification with an epsilon of {epsilon}")
    print(f"Using the bounding method: {method}") if PRINT else None

    df_results = pd.DataFrame(columns=RESULTS_COLUMNS)

    num_samples_verified = 0
    num_samples_correctly_classified = 0
    num_samples_safe = 0

    for dl_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Starting new batch") if PRINT else None

        if dl_idx == 0:
            if DEVICE == "cuda":
                inputs = inputs.cuda()
                targets = targets.cuda()
                model = model.cuda()
            print("Running on", DEVICE)

            lirpa_model = BoundedModule(
                model,
                torch.empty_like(inputs),
                device=inputs.device,
                verbose=True,
            )

        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
        ptb_inputs = BoundedTensor(inputs, ptb)

        preds = lirpa_model(ptb_inputs)
        pred_targets = torch.argmax(preds, dim=1)

        lb, ub = lirpa_model.compute_bounds(x=(ptb_inputs,), method=method.split()[0])
        if round_floats:
            preds = round_tensor(preds, decimals=5)
            pred_targets = torch.argmax(preds, dim=1)
            lb, ub = round_tensor(lb, decimals=5), round_tensor(ub, decimals=5)

        # Sanity check that bounds are indeed higher/lower than output
        assert (lb <= preds).all().item()
        assert (ub >= preds).all().item()

        attack_results = pgd(model, epsilon, inputs, targets, final_layer=False)
        # num_successful_attacks = sum(not value for value in attacks_results)

        # Iterate over each element in batch, first handling verification
        for i in range(len(targets)):
            # Each batch has len(inputs) images, and dl_idx indicates the batch number.
            # Therefore an MNIST image id would be `i + dl_idx * len(inputs)`
            new_row = {"mnist_id": i + dl_idx * len(inputs)}

            # num of samples verified
            num_samples_verified += 1

            (
                print(
                    f"Image {i} top-1 prediction is: {pred_targets[i]}, the "
                    f"ground-truth is: {targets[i]}"
                )
                if PRINT
                else None
            )

            classification_correct = (pred_targets[i] == targets[i]).item()

            num_samples_correctly_classified += 1 if classification_correct else 0

            # The `bound_sotmax` results in nan, using torch Softmax instead.
            lb_sm, ub_sm = bound_softmax(lb, ub, use_float64=True)
            # lb_sm, ub_sm = torch.nn.Softmax(dim=1)(lb), torch.nn.Softmax(dim=1)(ub)

            # Sanity check that post-softmax bounds are a valid probability, i.e. between 0 and 1
            assert 0 <= (lb_sm).all(), f"Lower bound lower than 0"
            assert (lb_sm).all() <= 1, f"Lower bound greater than 1"
            assert 0 <= (ub_sm).all(), f"Upper bound lower than 0"
            assert (ub_sm).all() <= 1, f"Upper bound greater than 1"

            # The model performs softmax in the last layer
            # softmax_preds = torch.softmax(preds[i], dim=0)
            # if round_floats:
            #     softmax_preds = round_tensor(softmax_preds, decimals=5)
            # assert approx_gte(softmax_preds, lb_sm[i], 1e-5), f"{dl_idx} {i}"
            # assert approx_lte(softmax_preds, ub_sm[i], 1e-5), f"{dl_idx} {i}"

            truth_idx = int(targets[i].item())

            # Check that the lower bound of the truth class is greater than
            # the upper bound of all other classes
            if (
                (
                    lb_sm[i][truth_idx]
                    > torch.cat(
                        (
                            ub_sm[i][:truth_idx],
                            ub_sm[i][truth_idx + 1 :],
                        )
                    )
                )
                .all()
                .item()
            ):
                classification_safe = True
            else:
                classification_safe = False

            # Keep track if the input was classified correctly and bounds were safe, but pgd attack was successful
            classification_correct_safe_pgd_attack_success = False

            if attack_results[i] and classification_safe:
                a, adv_output, adv_input = pgd(
                    model,
                    epsilon,
                    inputs,
                    targets,
                    final_layer=False,
                    return_model_output=True,
                )
                classification_correct_safe_pgd_attack_success = True
                if PRINT:
                    print(f"For the input {np.array(inputs[i,0,...])}")
                    print(f"We have the classification {targets[i,...]}")
                    print(f"We have the adv input {np.array(adv_input[i,0, ...])}")
                    print(f"We have the adv output: {adv_output[i,...]}")
                warnings.warn(
                    "Attack was successful but the bounds were not safe! Inputs. Model output: "
                    + str(torch.argmax(adv_output[i]).item())
                    + " Expected output: "
                    + str(targets[i].item())
                )

            for j in range(NUM_DIGIT_CLASSES):
                indicator = "(ground-truth)" if j == truth_idx else ""
                pred_indicator = "(prediction)" if j == truth_idx else ""
                safe_indicator = (
                    "(safe)" if j == truth_idx and classification_safe else ""
                )
                (
                    print(
                        "f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind} {pred} {safe}".format(
                            j=j,
                            l=lb_sm[i][j].item(),
                            u=ub_sm[i][j].item(),
                            ind=indicator,
                            pred=pred_indicator,
                            safe=safe_indicator,
                        )
                    )
                    if PRINT
                    else None
                )
                new_row[f"classification_{j}_lb"] = lb_sm[i][j].item()
                new_row[f"classification_{j}_ub"] = ub_sm[i][j].item()
                new_row["classification_prediction_idx"] = pred_targets[i].item()
                new_row["classification_target_idx"] = targets[i].item()
                new_row["classification_correct"] = classification_correct
                new_row["classification_correct_safe_pgd_attack_success"] = (
                    classification_correct_safe_pgd_attack_success
                )
            safe = (
                True
                if classification_safe
                and not classification_correct_safe_pgd_attack_success
                else False
            )
            new_row["classification_safe"] = safe
            if safe:
                num_samples_safe += 1

            new_row_df = pd.json_normalize(new_row)
            df_results = pd.concat([df_results, new_row_df], ignore_index=True)

    print(f"----\nSUMMARY\n----")
    print(f"For the method: {method}")
    print(f"Num samples verified: {num_samples_verified}")
    print(f"Num samples correctly classified: {num_samples_correctly_classified}")
    print(f"Num samples safe: {num_samples_safe}")
    print()

    results_summary = {
        "method": f"{method}",
        "num_samples_verified": f"{num_samples_verified}",
        "num_samples_correctly_classified": f"{num_samples_correctly_classified}",
        "num_samples_safe": f"{num_samples_safe}",
        "epsilon": f"{epsilon}",
    }

    filename = f"results_{method}_{epsilon}"
    filename += f"_rounded" if round_floats else ""
    save_results_to_csv(df_results, results_summary, filename)


def save_results_to_csv(results: pd.DataFrame, summary: dict, filename: str):
    """Save a Pandas Dataframe as CSV

    Args:
        results (pd.DataFrame): Pandas Dataframe to be saved as CSV
        summary (dict): Summary of the results to be saved as JSON
        filename (str): Name of the file - based on the method, epsilon value
    """
    results.to_csv(
        RESULTS_PATH / f"{filename}.csv",
        index=False,
    )

    with open(
        RESULTS_PATH / f"{filename}.json",
        "w",
    ) as file:
        json.dump(summary, file, indent=4)

    print(f"Saved to file { RESULTS_PATH / f'{filename}.csv'}")


def plot_MNIST_img(input: torch.Tensor, target: int):
    """Plot the MNIST image (for debugging purposes)

    Args:
        input (torch.Tensor): input image tensor
        target (int): target label value
    """
    pixels = np.array(input, dtype="uint8")
    pixels = pixels.reshape(28, 28)

    import matplotlib.pyplot as plt

    plt.title("Label is {label}".format(label=target))
    plt.imshow(pixels, cmap="gray")
    plt.show()


if __name__ == "__main__":
    cnn_network = MNIST_Net()

    model = get_mnist_deepproblog_model(cnn_network)

    test_dataset = AdditionDataset("test")
    # test_dataset = MNISTImages("test")

    # Confusion Matrix - commenting as it takes time to be computed
    # print(get_fact_accuracy(model, test_dataset, verbose=1))

    test_dataset_mnist = test_dataset.dataset
    # combine data and target into a single tuple, stacking 2 images and combining 2 targets to a list
    # test_dataset_mnist = tuple(zip(test_dataset_mnist.data.view(-1,1,28,28).to(torch.float32) / 255.0, test_dataset_mnist.targets))
    # imgs_targets = test_dataset.dataset
    # test_dataset_mnist = tuple((torch.stack((imgs_targets[i-1][0], imgs_targets[i][0])).float(), [imgs_targets[i-1][1], imgs_targets[i][1]]) for i in range(1, len(imgs_targets)))

    test_dl = TorchDataLoader(test_dataset_mnist, batch_size=BATCH_SIZE)

    print("Verifying CNN")

    # IBP bounds
    # calculate_bounds(cnn_network, test_dl, epsilon=0.1)
    calculate_bounds(cnn_network, test_dl, epsilon=0.01)
    calculate_bounds(cnn_network, test_dl, epsilon=0.001)
    calculate_bounds(cnn_network, test_dl, epsilon=0.0001)
    calculate_bounds(cnn_network, test_dl, epsilon=0.00001)

    # calculate_bounds(cnn_network, test_dl, epsilon=0.1, method="CROWN-IBP")
    # calculate_bounds(cnn_network, test_dl, epsilon=0.01, method="CROWN-IBP")
    # calculate_bounds(cnn_network, test_dl, epsilon=0.001, method="CROWN-IBP")
    # calculate_bounds(cnn_network, test_dl, epsilon=0.0001, method="CROWN-IBP")

    # calculate_bounds(cnn_network, test_dl, epsilon=0.1, method="CROWN")
    # calculate_bounds(cnn_network, test_dl, epsilon=0.01, method="CROWN")
    # calculate_bounds(cnn_network, test_dl, epsilon=0.001, method="CROWN")
    # calculate_bounds(cnn_network, test_dl, epsilon=0.0001, method="CROWN")
    # calculate_bounds(cnn_network, test_dl, epsilon=0.00001, method="CROWN")
