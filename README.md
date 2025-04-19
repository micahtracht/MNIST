# K-means from scratch on MNIST dataset
This project explores various different ways to classify digits on the MNIST dataset using various algorithms.

 ---

## Implemented from scratch (More added later):
 - **K means** 
     - Custom k-means implementation based on Euclidean distance.
     - Centroid updates, convergence, & visualization all done manually.
 - **Least Squares**
    - Implemented using a custom matrix class.
    - Pseudoinverse computed manually, with the exception of SVD, which numpy was used for. (argmax was also used for one-hot encoded labels and classification)

## Implemented w/ PyTorch:
 - **Convolutional Neural Network**
    - Built w/ `nn.Module`.
    - Uses 2 convolutional layers, ReLU, and max pooling: conv1 -> ReLU -> pool1 -> conv2 -> ReLU -> pool2 -> Flatten -> Fc1 -> ReLU -> fc2 -> logits

## Performance on test data
 - **CNN**: ~98.8% accuracy (5 epochs)
 - **Least-squares**: ~85% accuracy (when ran on all samples)
 - **K-means**: ~50% accuracy (varies based on clustering considerably)

## How to run (TBD)

Install dependencies:
```bash
pip install -r requirements.txt
```

Then run the main script, choosing which algorithm you want:
```bash
python main.py --mode cnn
python main.py --mode least_squares
python main.py --mode kmeans
```

---

## Directory overview
- `main.py`: Entry point for choosing models
- `cnn_runner.py`: CNN training & evaluation
- `kmeans_runner.py`: K-means clustering logic
- `least_squares_runner.py`: Least squares classifier
- `matrix.py`: Custom matrix operations used in least squares
- `CNN.py`: CNN class definition

---

## Coming soon (maybe):
 - Custom SVD implementation