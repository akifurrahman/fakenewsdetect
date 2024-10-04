# 📰 Fake News Detection Using Deep Learning

This project aims to build a **Fake News Detection System** using a **Convolutional Neural Network (CNN)**. The system classifies news articles as either **"Real"** or **"Fake"** based on their content. The project includes model development, dataset preprocessing, and deployment via a web interface.

## 🚀 Project Overview

In the era of rapid information dissemination, fake news has become a significant issue. This project tackles the challenge by implementing a machine learning model capable of distinguishing between real and fake news articles.

### ⚡ Key Features:
- **Data Preprocessing**: Cleaned and tokenized a dataset of over **10,000 articles** for model training.
- **Text Classification**: Built a **CNN model** using **Keras** and **TensorFlow** to classify articles as fake or real.
- **Web Application**: Deployed the model using **Flask**, allowing users to input text and receive real-time predictions.
- **Model Accuracy**: Achieved an impressive **99.36% validation accuracy** after 18 epochs of training.
- **User Interface**: Designed a simple UI where users can paste article content and the system will classify it as **Fake** or **Real**.

## 🛠️ Technologies Used:
- **Languages**: Python
- **Web Framework**: Flask
- **Machine Learning Libraries**: Keras, TensorFlow
- **Data Manipulation**: Pandas, NumPy
- **Natural Language Processing**: Tokenizer, Text Padding, Sequence Processing
- **Model Persistence**: **pickle** for saving model and tokenizer
- **Visualization**: Matplotlib, Seaborn

## 📑 Workflow:
1. **Data Preprocessing**: Cleaned data, handled missing values, and tokenized the text for model input.
2. **Model Development**: Implemented a **CNN model** with embedding layers, convolutional layers, and dropout regularization.
3. **Training**: Trained the model on **80% of the dataset**, validated on **20%**, and tested with **10%** of unseen data.
4. **Deployment**: Deployed the trained model in a Flask web app that classifies news articles as Fake or Real in real-time.

## 💻 How to Run the Project:
1. Clone the repository:
    ```bash
    git clone https://github.com/akifurrahman/fakenewsdetect.git
    ```
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the dataset and place it in the project directory.
4. Train the model using the provided Jupyter notebook, or use the pre-trained model (`model.h5`).
5. Run the Flask application:
    ```bash
    python app.py
    ```
6. Access the web app at `http://127.0.0.1:5000/`.

## 📊 Dataset:
The dataset used in this project contains news articles labeled as **Fake (1)** or **Real (0)**. The data includes titles, article content, and labels, providing a well-balanced distribution for training the model.

## 📈 Model Performance:
- **Training Accuracy**: 99.36%
- **Validation Accuracy**: 94.41%

### CNN Model Summary:
```
Layer (type)                Output Shape              Param #   
=================================================================
embedding (Embedding)       (None, 547, 11)           275011     
conv1d (Conv1D)             (None, 543, 16)           896        
dropout (Dropout)           (None, 543, 16)           0          
global_max_pooling1d (Glob  (None, 16)                0          
dropout_1 (Dropout)         (None, 16)                0          
dense (Dense)               (None, 8)                 136        
dense_1 (Dense)             (None, 1)                 9          
=================================================================
Total params: 276052
Trainable params: 276052
Non-trainable params: 0
```

## 🔗 Future Enhancements:
- Extending the model to work with multiple languages.
- Improving user interface for better user experience.

---

Feel free to customize the project and make it your own. Contributions and suggestions are always welcome!

---
