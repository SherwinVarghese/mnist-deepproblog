DeepProblog verification for the simple digit addition task

## Verification

### Setup

```bash
python3 -m venv venv
source ./venv/bin/activate
source .env
pip install poetry 
poetry install
```

### Background

Neuro-symbolic (Ne-Sy) models combine Neural Networks and Symbolic logic to develop efficient networks that can perform complex tasks using the outputs of several downstream networks.

DeepProbLog is a probabilistic logic programming language that enables the development of Ne-Sy models.

To ensure that the model is reliable and robust against small input perturbations, we use verification techniques to certify the neural networks. Thus, we can provide provable guarantees of network robustness for the Ne-Sy models.

The inputs classified as safe means that small changes in the input will not affect its classification.

## Neural Network Verification
[auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) is the state of the art verification tool that enables verification of the Neural Network against the properties under consideration.

Each input in the test dataset undergoes a perturbation `ε`. The perturbed inputs are then passed as inputs to the NN.

`auto_LiRPA` creates a wrapper around the NN to predict the approximate upper and lower bounds for the network.


- The code `mnist_deepproblog_verification.py` contains the method to verify a simple DeepProbLog MNIST network.
- After verifying each input against the perturbations, we obtain the upper and lower bounds. To perform a sanity check, we perform a Projected Gradient Descent (PGD) attack on the inputs. Then, we provide a warning if the model was classified as safe but attacks were successful. The source code for PGD attack is in the `pgd` file.

### Verifying for other Networks
In order to perform verification of other Neural Networks, the code in `mnist_deepproblog_verification.py` needs to be adapted.

1. `NUM_DIGIT_CLASSES` is the number of output classes, for MNIST, it is `10`. It has to be updated based on the NN
2. `RESULTS_COLUMNS` contains the input id and the lower and upper bound for each of the output classes. It also contains the classification target index of the input, the predicted classification index. Additionally it contains the boolean values if the classification is correct, classification is safe and if a PGD attack was successful on an input classified as safe.
3. The `PROGRAM_STRING` (specific to DeepProbLog) contains the predicate logic for the symbolic component, that gets converted into probabilistic logic.
4. The verification is driven by the function `calculate_bounds()` that takes the Neural network model, the data loader, `ε` and the `method`, which denotes the verification method. The default verification method is Interval Bound Propagation (`IBP`), which provides the loosest bounds, but is computationally fast. 
5. `auto_LiRPA` supports the following methods in addition to `IBP`: 
    * `IBP`: purely use Interval Bound Propagation (IBP) bounds.
    * `CROWN-IBP`: use IBP to compute intermediate bounds,
    but use CROWN (backward mode LiRPA) to compute the bounds of the
    final node.
    * `CROWN`: purely use CROWN to compute bounds for intermediate
    nodes and the final node.
    * `CROWN-Optimized` or `alpha-CROWN`: use CROWN, and also optimize the linear relaxation parameters for activations.