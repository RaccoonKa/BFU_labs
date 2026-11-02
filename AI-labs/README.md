# AI-labs

Practical lab assignments for an introductory AI and machine learning course. The repository walks through the core ideas of modern ML — from implementing a single artificial neuron by hand to convolutional networks for image classification and LSTMs for text generation — using NumPy, scikit-learn, PyTorch, and TensorFlow/Keras.

## What the course is about

Artificial intelligence is built on models that learn patterns from data instead of being explicitly programmed. This course follows the path from the simplest possible model to the architectures behind modern vision and language systems: implementing the basics from scratch, comparing them against library implementations, and analyzing *why* one approach outperforms another.

**Linear regression from scratch.** Regression is the entry point to machine learning. A linear model fits a line (or a hyperplane) to the data by minimizing squared error, which can be solved analytically or via gradient descent. The labs compare a hand-written implementation against scikit-learn's `LinearRegression`, visualize the regression line and squared-error boxes, and evaluate results with MSE, MAE, R² and MAPE.

**Neural networks from scratch.** An artificial neuron applies a weighted sum followed by a sigmoid activation. Stacking neurons into layers and training them with backpropagation and gradient descent demonstrates the core machinery behind every deep model. On the Iris dataset, a single neuron and a two-layer network (10 neurons per layer) are compared — both achieve perfect accuracy because the chosen features are linearly separable, so the deeper network's extra capacity collapses to the same linear boundary.

**MLP vs CNN on MNIST.** Fully-connected networks flatten an image into a vector and lose spatial structure. Convolutional networks (LeNet-style) keep the 2D topology, learn local patterns (edges, loops, curves) through shared convolution kernels, and are more robust to shifts and distortions. The comparison shows CNN outperforming MLP (98.30% vs 97.69%) and explains why.

**LSTM text generation.** Recurrent networks model sequences character by character, capturing order and context. A stacked LSTM trained on a small text corpus shows excellent training convergence (loss drops from 3.18 to 0.0067 in 20 epochs) — but on a tiny dataset this is pure memorization, and the generated output degenerates into "abracadabra": the model has only learned primitive character-level patterns, not words or sentences.

**Architectures and trade-offs.** From analytic solutions to gradient descent, from single neurons to deep nets, from dense layers to convolutions, and from feedforward to recurrent models — each step adds expressive power at the cost of data and computation requirements.

## Course structure

| # | Topic | Key tools |
|---|-------|-----------|
| 1 | Artificial neuron from scratch, backpropagation, Iris classification | `numpy`, `sklearn.datasets`, `matplotlib` |
| 2 | MNIST: MLP vs CNN (LeNet) — metrics comparison | `sklearn.neural_network`, `torch`, `torchvision` |
| 3 | Linear regression: manual vs scikit-learn, metrics (MSE, MAE, R², MAPE) | `numpy`, `sklearn.linear_model`, `pandas` |
| 4 | LSTM character-level text generation | `tensorflow.keras`, `numpy` |

<div align="center">  
  <img src="https://user-images.githubusercontent.com/114250490/192084774-af22327e-799c-41e0-87bd-4df4842b6a6f.gif" width="400" />
</div>  
