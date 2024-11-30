# Fine-Tuned Model Prediction: Product Price Prediction

Welcome to the repository! This project demonstrates how to fine-tune an open-source model to predict product prices and outperform current models like GPT-4. We've leveraged a smaller 8 billion parameter model and fine-tuned it specifically for this task, which allowed us to achieve better-than-expected results.

## Overview

In this project, we focused on fine-tuning an open-source language model to predict product prices. The main objective was to enhance its performance on a dataset of 251 products and achieve accuracy that beats traditional machine learning models as well as GPT-4, which is a large-scale frontier model.

## Achievements

### Key Results:
- **Accuracy**: The model achieved an impressive **$47** error margin on the test set, outperforming traditional models and GPT-4.
- **Comparison to Other Models**:
    - **Traditional Machine Learning (Random Forest)**: $97 error.
    - **Human**: $127 error.
    - **GPT-4**: $76 error.
    - **Fine-Tuned Model**: $47 error.
- The fine-tuned model was able to closely match the ground truth (prices) when compared to a traditional baseline model. This is especially impressive given the variability in product prices.

### Performance:
- The model achieved a **46.67% accuracy**, which is a fantastic result considering the variability inherent in predicting product prices. 
- This was achieved using an **8 billion quantized model** with targeted fine-tuning for product price prediction.

### Green and Yellow Results:
- The model displayed a series of **green and yellow results**, indicating the model’s ability to make reliable predictions while acknowledging its limitations (shown by the occasional red results).

### Outperformed GPT-4:
- The model exceeded GPT-4's prediction capabilities on this specific task, demonstrating the power of fine-tuning a smaller open-source model for specialized tasks.

## Challenge: Improve the Model

While the fine-tuned model performed very well, there is always room for improvement. Here's a breakdown of areas where enhancements can be made:

1. **Hyperparameter Optimization**: 
    - Experiment with different hyperparameters such as learning rate, batch size, and optimizer choices (e.g., AdamW, SGD).
    - Use **Weights & Biases** for experiment tracking and to find the optimal hyperparameters.

2. **Inference Techniques**:
    - Explore ways to improve the model during inference time. This may involve adjusting how we process the input data or making inference faster and more accurate.

3. **Data Curation**:
    - Revisit the dataset for better data organization and prompt engineering. The quality and structure of the input data can significantly impact the prediction accuracy.

4. **Alternative Models**:
    - Test other models like **Jama**, **Kwon**, or **Fi3** to see how they compare to the current fine-tuned model.
    - Explore a version of **Llama 3** quantized to 8-bits instead of 4-bits for better performance.

5. **Additional Models**:
    - Experiment with larger models, such as a **14 billion parameter version** of Fi3, to see if it can further improve the predictions.

## Final Objective

The ultimate goal is to push the error margin to **below $40**. The current model has an error of $47, which is already impressive, but it is expected that a refined version will outperform this.

### Next Steps:
- Experiment with hyperparameters, different models, and other fine-tuning techniques to bring the model closer to the $40 target.
- Track progress by comparing the updated results against the current model to see the impact of changes.

## Files and Notebooks

This repository contains the following files and notebooks that guide the process of training, testing, and evaluating the fine-tuned model:

1. **QLora_one (1).ipynb**: A notebook for training the model using **QLora**.
2. **Testing_our_Fine_tuned_model.ipynb**: The notebook for testing the performance of the fine-tuned model and evaluating accuracy.
3. **Training.ipynb**: The notebook that covers the overall training process, fine-tuning techniques, and the results comparison.

## Conclusion

This project demonstrates the ability to fine-tune an open-source model to outperform even frontier models like GPT-4 in a specific task. The next step involves further fine-tuning, model experimentation, and optimizing the error margin to below $40. We invite you to join the challenge and help improve this model
