# Sentiment-Analysis-of-IMDb-Movie-Reviews-Using-BERT

## Problem Solved

This project aims to build a binary sentiment analysis model using the IMDb movie reviews dataset. This dataset contains 50,000 reviews, labeled as either positive or negative, with an even number of highly polarizing reviews in both categories. Sentiment analysis is challenging because it requires understanding the contextual meaning of the text to determine its sentiment.

## How it is Solved

Solve the problem by leveraging **BERT (Bidirectional Encoder Representations from Transformers)**, a pre-trained state-of-the-art model for NLP tasks. The BERT model is fine-tuned for sentiment classification on the IMDb dataset. Here are the key steps:

- **Dataset**: The [IMDb movie reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download) is used, with 25,000 reviews for training and 25,000 reviews for testing.
- **Preprocessing**: Text is tokenized into word IDs and padded to a maximum sequence length of 128 tokens using BERT’s preprocessing model from TensorFlow Hub.
- **Model**: We fine-tune the **Small BERT** model (BERT-Base Uncased) from TensorFlow Hub.
- **Training**: The model is trained for 5 epochs using the AdamW optimizer with learning rate decay.
- **Validation**: A validation set was created by splitting the training data (80% train, 20% validation).

## Solution Overview

1. **Preprocessing**: The text is tokenized and transformed into BERT input IDs, input masks, and segment IDs using a preprocessing model.
2. **Model Architecture**:
   - **BERT Layer**: BERT (small_bert/bert_en_uncased_L-2_H-128_A-2) was loaded from TensorFlow Hub.
   - **Classification Head**: Added a dense layer for binary classification (positive/negative).
   - **Optimizer**: AdamW optimizer with learning rate decay.
   - **Loss Function**: Binary Cross-Entropy for sentiment classification.
3. **Training**: Model is fine-tuned on 20,000 training samples and validated on 5,000 samples, with early stopping to avoid overfitting.
4. **Testing**: Evaluated the trained model on the 25,000 test reviews.

## Outcome Results

### Quantitative Results

- **Training Accuracy**: 76.88%
- **Validation Accuracy**: 76.56%
- **Test Accuracy**: 76.57%
- **Test Loss**: 0.466

### Impact

The fine-tuned BERT model provides accurate sentiment predictions for movie reviews. Here are some example results from the model:

- _Input: "This is such an amazing movie!"_  
  - Predicted Score: **0.863** (Positive)

- _Input: "The movie was terrible..."_  
  - Predicted Score: **0.073** (Negative)

### Inference Results

The model correctly identifies the sentiment of unseen movie reviews, demonstrating its effectiveness for sentiment classification.

## Conclusion

This project demonstrates the power of BERT for sentiment classification. By fine-tuning a pre-trained BERT model, we achieved a high accuracy of 76.5% on the IMDb movie reviews dataset. BERT's ability to capture contextual meaning enables it to perform well on nuanced sentiment analysis tasks.

## Technologies Used

- **Python** for coding
- **TensorFlow** and **Keras** for model training and evaluation
- **TensorFlow Hub** for loading pre-trained BERT models
- **IMDb Movie Reviews Dataset** for training and evaluation

## Future Work

- Experiment with larger BERT models (e.g., **BERT-Large**).
- Improve accuracy by adjusting hyperparameters or using different optimizers.
- Apply the model to other text classification problems, such as multi-class sentiment analysis.

