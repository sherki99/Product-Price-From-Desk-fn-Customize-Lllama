# Fine-Tuned Product Price Prediction Model

This repository showcases the work of fine-tuning an open-source language model to predict product prices, achieving outstanding performance on a test set. By leveraging a smaller, 8 billion parameter model and fine-tuning it for this specific task, the project demonstrates how tailored models can outperform even larger frontier models like GPT-4 in certain domains.

## Overview

In this project, we fine-tuned an open-source model to predict the prices of products, achieving an impressive performance with a **$47** error margin on the test dataset of 251 products. This fine-tuned model was compared to traditional machine learning models, a human baseline, and GPT-4, showcasing how smaller models can outperform current large-scale models when applied to specialized tasks.

### Key Results:
- **Fine-Tuned Model Accuracy**: The model achieved a **$47 error** on the test set.
- **Comparison to Other Models**:
    - **Traditional Machine Learning (Random Forest)**: $97 error.
    - **Human**: $127 error.
    - **GPT-4**: $76 error.
    - **Fine-Tuned Model**: $47 error.
- **Performance**: The fine-tuned model was able to closely predict product prices and demonstrated reliability across multiple test samples.

### Metrics:
- **Model Accuracy**: 46.67% accuracy.
- **Prediction Quality**: The model performed well on a variety of predictions, with results showing a mix of green (accurate) and yellow (less accurate) predictions, reflecting the inherent variability of product prices.

## Model Details

This fine-tuned model was built by starting with an **open-source 8 billion parameter model**, which was then adapted for the task of price prediction. The model was trained on a dataset that included product pricing information, and fine-tuning techniques were applied to improve its performance on predicting prices for new, unseen products.

### Training Process:
- **Data**: A dataset of 251 products was used for training and testing, with the goal of predicting product prices.
- **Fine-Tuning**: Techniques such as adjusting learning rates, optimizer choices, and batch sizes were explored to achieve the best performance.
- **Evaluation**: The model was evaluated on accuracy and error margin, where it showed an exceptional ability to predict product prices, outperforming traditional machine learning models and even large-scale models like GPT-4 for this specific task.

## Results

The fine-tuned model demonstrated significant improvements over baseline models:
- **Traditional models**: Such as Random Forests, achieved an error margin of $97.
- **GPT-4**: The frontier model reached an error margin of $76.
- The fine-tuned open-source model achieved an impressive **$47 error**, making it the most accurate model for this task.

## Files and Notebooks

This repository contains the following files and notebooks:

1. **QLora_one (1).ipynb**: The notebook where the model is trained using the **QLora** technique.
2. **Testing_our_Fine_tuned_model.ipynb**: A notebook that tests the fine-tuned model, evaluates its accuracy, and compares predictions to the ground truth.
3. **Training.ipynb**: This notebook covers the overall training process, including the fine-tuning steps and configuration settings for the model.

## Conclusion

By fine-tuning a smaller open-source model for the specific task of predicting product prices, we have achieved significant improvements over both traditional machine learning models and frontier models like GPT-4. The project demonstrates the effectiveness of specialized fine-tuning and highlights the potential for open-source models to outperform larger models when trained for specific use cases.

This is a great demonstration of how model fine-tuning and tailored approaches can lead to industry-leading results in specialized domains. The fine-tuned model for product price prediction is a significant step forward in the use of open-source models for real-world applications.
