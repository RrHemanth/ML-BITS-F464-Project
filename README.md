# Machine Learning Assignments (BITS F464)

Coursework for the Machine Learning course at BITS Pilani. The repo has two assignments:

1. **Tumor classification.** Three classifiers written from scratch with NumPy (Fisher's linear discriminant, a perceptron, logistic regression), applied to the Wisconsin breast cancer data.
2. **Income prediction and digit recognition.** A from-scratch Naive Bayes compared with KNN and logistic regression on census data, then a neural network on MNIST.

The goal throughout was to understand each algorithm by building it, then check how it behaves on real data.

## Repo layout

```
Assignment 1 - Tumor Classification/
    Dataset/Dsata Set.csv            569 tumors, 30 features, diagnosis label
    ML_LDA.ipynb                     Fisher's linear discriminant
    Perceptron_ML.ipynb              Perceptron
    ML_Logistic.ipynb                Logistic regression (batch, mini-batch, stochastic)

Assignment 2 - Income Prediction+Handwritten digit classification/
    Dataset_IncomePrediction/adult.csv   Census income data (32,561 rows)
    naive_bayes_ML.ipynb             Naive Bayes from scratch
    KNN_ML.ipynb                     KNN and logistic regression (scikit-learn)
    ANN_ML.ipynb                     Neural network on MNIST
    ML-Assignment_2-Report.pdf       Write-up with the full experiment comparison
```

---

## Assignment 1: Tumor classification

### The data

The Wisconsin breast cancer data: 569 tumors described by 30 measurements (mean, standard error and worst value of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension). Each tumor is labelled malignant (212) or benign (357).

A few cells were missing, in six columns: `radius_mean`, `perimeter_mean`, `concavity_mean`, `compactness_se`, `area_worst` and `concavity_worst`. Two ways of handling them were tried: dropping those rows, and filling with the column mean. The labels were encoded as numbers (malignant = 1 for the discriminant and perceptron, with -1 for benign in the perceptron).

For all three models the first rows are used for training and the later rows for testing, with no shuffling between the two.

### Fisher's linear discriminant (`ML_LDA.ipynb`)

Fisher's method looks for the single direction along which the two classes are best separated, then classifies by where a tumor lands on that line. The notebook builds it in steps:

1. Compute the mean of each class and the difference between them.
2. Compute each class's scatter matrix and add them to get the within-class scatter.
3. Take the direction `w = pinv(S_w) · (mean_0 - mean_1)` and normalize it to length 1.
4. Project every tumor onto `w`, giving one number per tumor.
5. Fit a bell curve to each class's projected values and solve for the point where the two curves cross. That point is the decision threshold.
6. Plot the projected points, both curves and the threshold, then classify the test tumors by which side they fall on.

It was run on two setups: the first with a 378 / 186 train/test split, and a second that drops `texture_mean` and trains on fewer rows.

### Perceptron (`Perceptron_ML.ipynb`)

The classic perceptron: start with all-zero weights, and for each training tumor that is classified wrongly, add `label × features` to the weights. This is repeated for 10,500 passes through the training rows. A tumor is predicted malignant if its dot product with the weights is positive.

Four variations were tried, to see how sensitive the perceptron is to the data it sees:

- **PM1:** raw features, original row order.
- **PM2:** training rows shuffled.
- **PM3:** features standardized (z-score).
- **PM4:** the order of the feature columns shuffled.

### Logistic regression (`ML_Logistic.ipynb`)

Three versions of logistic regression were written from scratch, all using a sigmoid and cross-entropy cost with gradient descent:

- **Batch gradient descent:** one update per pass over the data.
- **Mini-batch gradient descent:** the data is shuffled each pass and updated in batches of 32.
- **Stochastic gradient descent:** the same code with a batch size of 1.

The notebook first prepares the tumor data (both missing-value approaches, then z-scoring and a 67 / 33 split), then reuses the same classes on the income data from Assignment 2. On the income data, batch gradient descent (1,000 iterations, learning rate 0.0001) reached 75.5% accuracy, with the cost falling slowly from 0.548 to about 0.54. The learning rate was small, so it had barely converged.

### Results on the tumor data

Malignant is the positive class, so recall is the share of malignant tumors caught.

| Model | Accuracy | Recall | Precision |
|---|---|---|---|
| Fisher's linear discriminant | 98.8% | 98.3% | 96.7% |
| Perceptron | 93.3% | 92.3% | 80.0% |

- The discriminant misclassifies only 3 test tumors: it misses 1 malignant tumor and raises 2 false alarms.
- The perceptron, on the 186 test tumors, catches 36 of 39 malignant ones and raises 9 false alarms.

Recall matters most here, since a missed malignant tumor costs more than a false alarm. The discriminant is the stronger model, and a plain linear method already separates this data well.

---

## Assignment 2: Income prediction and digit recognition

### Part A: Predicting income (UCI Adult data)

**The data.** About 32,500 people with 14 features (age, work class, education, marital status, occupation, hours per week and so on), and a target: income above or below $50K. The CSV has no header, so columns are named A to O in the notebooks.

**Preparation.**
- Missing values (marked `?`) appear in work class, occupation and native country. These were filled with the most common value in each column.
- Text categories were converted to numbers with a small label-encoding function.
- Features were standardized with z-scores.
- The first 67% of rows were used for training and the last 33% for testing.

**Models.**
- **Naive Bayes, from scratch (`naive_bayes_ML.ipynb`):** a `NaiveBayes` class that computes class priors and per-feature likelihoods, with an `alpha` parameter for Laplace smoothing, and picks the class with the highest posterior. It also has its own accuracy, precision, recall and F1 methods. This version drops two columns (C and J).
- **K-nearest neighbors (`KNN_ML.ipynb`):** scikit-learn's classifier with k = 239. The report explains the choice: the square root of the data size is about 180, and values around it were tried, with 239 giving the best accuracy.
- **Logistic regression (`KNN_ML.ipynb`):** scikit-learn's version, on the same split.

**Results (from the report).**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 81.1% | 64.8% | 50.8% | 0.570 |
| KNN (k = 239) | 82.5% | 71.4% | 47.8% | 0.573 |
| Logistic regression | 83.7% | 73.2% | 53.3% | 0.617 |

What stood out:
- Logistic regression came out ahead in the report, and KNN beat Naive Bayes on accuracy and precision.
- Laplace smoothing raised Naive Bayes accuracy from about 80% to 82%.
- Over 10 random train/test splits, Naive Bayes accuracy stayed near 81%, and recall and F1 moved around more than accuracy and precision.
- Recall sits near 50% for all three models. Most people in the data earn under $50K, so the models miss many of the high earners.

The notebooks in the repo were left at a different run from the report, so rerunning them gives slightly different numbers. KNN and logistic regression come out around 82.6% and 82.7% accuracy, close to a tie, and the Naive Bayes notebook scores lower than the report's figure.

### Part B: Handwritten digits (MNIST)

A fully connected neural network in Keras on MNIST: 60,000 training images and 10,000 test images of digits, each 28 × 28 pixels. Pixel values are scaled to 0-1 and each image is flattened into 784 inputs.

**The notebook's network (`ANN_ML.ipynb`).** Two hidden layers of 512 and 256 units with tanh, and a 10-unit sigmoid output layer. It is trained with Adam on sparse cross-entropy for 5 epochs.

- Training accuracy climbs from 96.8% to 98.8% over the 5 epochs.
- Test accuracy is **97.6%**.
- The confusion matrix shows where it slips. The most common mistakes are 6 read as 4 (17 images), 5 read as 3 (14), 7 read as 2 (13) and 4 read as 9 (11).

**The report's architecture comparison.** The report goes further and compares 15 networks with 2 or 3 hidden layers of 100 to 150 neurons and different mixes of ReLU, tanh and sigmoid.

- Test accuracy ranged from 97.1% to 98.0%.
- The best was three hidden layers (100, 150, 100) with ReLU, at 98.0%.
- A one-sample t-test of the 15 accuracies against 97.5% found the average (97.6%) not significantly different. So no architecture stood out, and at this size the choice of layers and activations barely mattered.

---

## What we took from it

- On the tumor data, simple linear methods are very strong, and the discriminant beat the perceptron because it uses the class spread instead of reacting to individual mistakes.
- The perceptron and logistic regression are sensitive to scaling and learning rate. Features on different scales slow learning, and too small a learning rate leaves gradient descent far from converged.
- On the income data, the models trade precision against recall, and accuracy alone hides the fact that the minority class is poorly caught.
- For MNIST, a plain network already reaches about 97 to 98%, and tuning its shape gave very little beyond that.

## Running it

Install `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` and `tensorflow`, then open any notebook in Jupyter and run it top to bottom. A few notes:

- The notebooks read their CSV from the current folder, so check the file path in the data-loading cell. The tumor CSV is named `Dsata Set.csv` in the repo, and `ML_Logistic.ipynb` looks for a differently named file.
- `ML_Logistic.ipynb` also uses `adult.csv` for its second half.
- The ANN notebook downloads MNIST through Keras the first time it runs.
