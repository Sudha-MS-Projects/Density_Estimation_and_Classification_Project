# Density Estimation and Classification Project
This project is about implementing a two-class Naïve Bayes classifier for digit classification (digit ‘0’ and digit ‘1’) on a subset of the MNIST dataset. The goal is to perform the following tasks: 
* Feature extraction
* Parameter calculation
* Naïve Bayes classifier implementation
* Label prediction
* Accuracy evaluation.

## Dataset
The dataset used in this project is a subset of the MNIST dataset, which contains handwritten images of digits ‘0’ and ‘1’.

### Dataset Details:
**Training Set:**
* Digit '0' : 5000 samples
* Digit '1' : 5000 samples
  
**Test Set:**
* Digit '0' : 980 samples
* Digit '1' : 1135 samples

The prior probabilities for both digits are assumed to be equal: P(Y = 0) = P(Y = 1) = 0.5.

### Data Files
The dataset is provided as .mat files, containing the images for digit "0" and "1".
* train0.mat: Training data for digit "0"
* train1.mat: Training data for digit "1"
* test0.mat: Test data for digit "0"
* test1.mat: Test data for digit "1"

### Data Generation Process
To ensure uniqueness and reproducibility, we generate a subset of data files from the provided MNIST dataset based on student ID.

Example: If student ID ends with 1268, the following files will be generated:
* digit0_stu_train1268.mat (training data for digit 0)
* digit1_stu_train1268.mat (training data for digit 1)
* digit0_testset.mat (test data for digit 0)
* digit1_testset.mat (test data for digit 1)

These files are loaded into the program using scipy.io.loadmat() in Python.

## Tasks Overview
### Task 1: Feature Extraction
In this task, we extract two features from training datasets.

* **Feature1:** Average brightness - the mean of pixelvalues in each image.
* **Feature2:** Standard brightness - the standard deviation of pixel values in each image.

These features are calculated from the original 28x28 pixel images and are stored as 1x2 arrays, where each column represents a feature.

### Task 2: Naïve Bayes Parameter Calculation
In this task, we calculate 8 parameters required for the Naïve Bayes classifier:
1. Mean of Feature 1 for digit "0"
2. Variance of Feature 1 for digit "0"
3. Mean of Feature 2 for digit "0"
4. Variance of Feature 2 for digit "0"
5. Mean of Feature 1 for digit "1"
6. Variance of Feature 1 for digit "1"
7. Mean of Feature 2 for digit "1"
8. Variance of Feature 2 for digit "1"

These parameters are used to calculate the likelihood of an image belonging to each class (digit "0" or digit "1").

### Task 3: Naïve Bayes Classifier Implementation and Prediction
Using the parameters from Task 2, we implement the Naïve Bayes classifier to predict the labels for the test dataset. The classification formula is applied to the test data to assign labels of either digit "0" or digit "1" based on the calculated probabilities.

#### Naïve Bayes Classifier Overview
The Naïve Bayes classifier is based on Bayes Theorem, which calculates the probability of a class given the observed features. The classifier assumes that the features are conditionally independent given the class, which is why it is called "Naïve".

For a given class 𝐶, Bayes' Theorem states:

<img width="191" alt="image" src="https://github.com/user-attachments/assets/2d594a4f-6814-4f85-bf65-88da8b22f7ab" />

Where:
* 𝑃(𝐶∣𝑋) is the posterior probability of class 𝐶 given the features 𝑋.
* 𝑃(𝑋∣𝐶) is the likelihood of observing the features 𝑋 given the class 𝐶.
* 𝑃(𝐶) is the prior probability of class 𝐶.
* 𝑃(𝑋) is the evidence or normalization factor.

The goal of classification is to predict the class 𝐶 that maximizes the posterior probability 𝑃(𝐶∣𝑋). Since 𝑃(𝑋) is constant for all classes, we only need to maximize 𝑃(𝑋∣𝐶)𝑃(𝐶).

Since the features are assumed to be independent, we use a **Gaussian distribution** to model the likelihood of eacg feature.
For each feature 𝑋𝑖​ in the vector 𝑋, the likelihood is given by:

<img width="283" alt="image" src="https://github.com/user-attachments/assets/3528db01-30aa-46e0-a68b-c930bc0bb293" />

Where:
* $\mu_i$ is the mean of feature 𝑖 for class 𝐶.
* $\sigma^2_i$ is the variance of feature 𝑖 for class 𝐶.

The total likelihood for a 2D feature vector 𝑋 = (𝑋1, 𝑋2) is:

<img width="218" alt="image" src="https://github.com/user-attachments/assets/b912f2d9-4afb-4bbe-90bc-d2738810e862" />

#### Naïve Bayes Classifier Formula
The Naïve Bayes classifier formula computes the posterior probability for each class as:

<img width="262" alt="image" src="https://github.com/user-attachments/assets/adcd124b-7b3e-4a4b-a9de-b911549a5010" />



**Implementation Steps:**

For each test data point, calculate the likelihood for both classes using the Gaussian formula. Multiply the likelihood by the prior probability (which is 0.5 for both classes) and select the class with the highest probability.

### Task 4: Accuracy Calculation
After making predictions, we evaluate the performance of the Naïve Bayes classifier by calculating the accuracy, which is the percentage of correct predictions.

<img width="344" alt="image" src="https://github.com/user-attachments/assets/5a1c8088-0619-4588-a34a-5507b6eb1ab6" />

## How to Run the Code
### Prerequisites
* Python 3.x
 
### Libraries and Dependencies
The following Python libraries are required to run the project:
* numpy: For numerical computations and handling arrays.
* scipy: For reading the .mat files (MATLAB data format).
* Jupyter Notebook (Optional, if running via Jupyter)
  
You can install the required libraries using **pip**:

    pip install numpy scipy jupyter

## Steps to Run the Project
### 1. Clone the Repository
Clone the repository to your local machine using the following command:

    git clone https://github.com/Sudha-MS-Projects/Density_Estimation_and_Classification_Project.git

### 2. Run the Jupyter Notebook
Navigate to the project directory and open the Jupyter Notebook:
  
    jupyter notebook

This will open the notebook interface in the browser. Find and open the Density_Estimation_and_Classification.ipynb notebook and run the cell.

### 3. Review the Result
Once the notebook or script runs successfully, you will see:
* Parameters for The Naive Bayes classifier (mean and variance for each feature).
* The final accuracy of the classifier based on the test set.
