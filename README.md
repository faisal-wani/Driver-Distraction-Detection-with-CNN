# Driver Distraction Detection with CNN
This repository contains the code and resources for a deep learning model built using Convolutional Neural Networks (CNN) to detect distracted driver behaviors. The model is trained to classify various distracted driving activities from labeled images of drivers, helping to enhance road safety by detecting and mitigating behaviors such as texting, drinking, or reaching back.
## Table of Contents
    Project Overview
    Dataset
    Installation and Setup
    Model Architecture
    Training and Evaluation
    Results
    Usage
    Acknowledgements
## Project Overview
The primary goal of this project is to create an image classification model capable of accurately identifying distracted driving behaviors based on visual inputs. This model can serve as a foundation for applications that monitor driver behavior and provide real-time alerts, potentially reducing accidents caused by distracted driving.

### The model is trained to recognize five types of driver behavior:

Not distracted
Texting (Right Hand)
Calling (Right Hand)
Drinking
Reaching Back Seat
## Dataset
The dataset used in this project, StateFarm Driver Distraction Dataset (StateFarmDDD), consists of images categorized into folders, with each folder representing a specific driver behavior class. The images are in .png and .jpg formats.
### Dataset Structure:
Before running the model, ensure that the dataset is organized in the following structure:

StateFarmDDD
├── c0_NotDistracted
├── c1_TextingRightHand
├── c2_CallingRightHand
├── c6_Drinking
└── c7_ReachingBackSeat

## How to Download:
You can download the dataset from Kaggle:
https://www.kaggle.com/datasets/faisalwani50/statefarm

Once downloaded, extract the contents and place the dataset in the data_folder of your project directory.


## Installation and Setup
#### 1.Clone the repository:
        bash
        Copy code
        git clone https://github.com/faisal-wani/Driver-Distraction-Detection-with-CNN.git
        cd Driver-Distraction-Detection-with-CNN

#### 2.Install the required libraries:
    This step ensures that all the necessary libraries for the project are installed. The libraries include:

        numpy: for numerical operations
        pandas: for data manipulation
        tensorflow: for deep learning
        scikit-learn: for machine learning tasks
        matplotlib: for plotting and visualizations
        opencv-python: for image processing
        
        You can install these libraries using pip in your terminal or command prompt:
        
        bash
        Copy code
        pip install numpy pandas tensorflow scikit-learn matplotlib opencv-python
### 3.Run the Jupyter notebook:

        bash
        Copy code
        jupyter notebook driver-distraction-detection-ipynb.ipynb

## Model Architecture

This CNN-based model is implemented using TensorFlow and Keras, featuring the following layers:

Two Convolutional Layers (Conv2D) with MaxPooling for feature extraction
Dropout Layer for regularization
Flatten Layer to reshape data for the Dense layers
Fully Connected Dense Layers for classification
Softmax Output Layer for multi-class classification

## Training and Evaluation
##### Data Processing
          Image Resizing: All images are resized to (224, 224) pixels.
          Normalization: Pixel values are scaled between 0 and 1.
          Train-Test Split: The dataset is divided as:
                              Training Set: 80%
                              Testing Set: 20%
#### Compilation and Training:
          Loss Function: Categorical Cross-Entropy
          Optimizer: Adam
          Metrics: Accuracy
          Epochs: 5

## Results
After training, the model achieved remarkable accuracy on the test data. Below are the key metrics and results:

#### Accuracy on Test Data: 99.8%
Confusion Matrix: A visual representation of the model’s performance across different classes is available in the notebook.

## Model Summary:
Layer                              	OutputShape                                       	Parameters
Conv2D	                          (None, 222, 222, 32)          	                         896
MaxPooling2D	                    (None, 111, 111, 32)	                                    0
Conv2D	                          (None, 109, 109, 64)	                                  18,496
MaxPooling2D	                    (None, 54, 54, 64)	                                      0
Dropout	                          (None, 54, 54, 64)                                        0
Flatten                          	(None, 186624)	                                          0
Dense	                            (None, 128)	                                          23,888,000
Dense	                            (None, 64)	                                            8,256
Dense	                               (None, 5)	                                           325
Total Parameters: 23,915,973
Trainable Parameters: 23,915,973

## Usage
To use this project to detect driver distractions:
  1.Download and set up the dataset in the specified format.
  2.Run the code in the Jupyter notebook.
  3.Visualize sample predictions with actual labels and predicted classes, as shown in the notebook.
The model can be further customized and deployed for real-time applications, using video processing tools like OpenCV.

## Acknowledgements
StateFarmDDD Dataset: Thanks to Kaggle for providing the dataset.
Kaggle: Computational resources were provided by Kaggle, allowing us to develop and train this model effectively.
