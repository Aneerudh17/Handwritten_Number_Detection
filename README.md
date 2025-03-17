# Handwritten Number Detection
![alt image](https://github.com/Aneerudh17/Handwritten_Number_Detection/blob/main/Test_Accuracy.png)
![alt image](https://github.com/Aneerudh17/Handwritten_Number_Detection/blob/main/prediction1.png)
![alt image](https://github.com/Aneerudh17/Handwritten_Number_Detection/blob/main/Prediction2.png)
![alt image](https://github.com/Aneerudh17/Handwritten_Number_Detection/blob/main/prediction3.png)
### Features:
- Data preprocessing: Clean and prepare the dataset for model training.
- Model creation: Build a neural network model using TensorFlow.
- Training: Train the model using the prepared dataset.
- Evaluation: Test the model’s performance using a separate test set.

## Data Used

- **Dataset:** The project uses a MNIST dataset (e.g., classification or regression task, depending on your case).
- **Training Data:** 60000 images were used for training the model.
- **Test Data:** : 10000 images were used to evaluate the performance of the model.
- **Preprocessing:** The data is preprocessed (e.g., normalization, splitting into training and test sets) before feeding it into the model.

## Requirements

- Python 3.9.0 (Not compatible with version above 3.9, source: ive tried myself)
- TensorFlow 2.19.0
- NumPy (compatible version)
- Pandas (if necessary, depending on data format)
- Matplotlib or other libraries (for visualizations)

### Installing the Requirements
(optional, ive created a virtual environment to install the necessary modules to avoid issues)
1. Create a virtual environment:
   ```bash
   python -m venv .venv
OR
You can simply install it using pip: (make sure to upgrade pip to install the latest release of the individual packages)
  ```bash
python -m pip install tensorflow matplotlib numpy
