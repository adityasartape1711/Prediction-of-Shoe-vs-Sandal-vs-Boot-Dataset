
## Footwear Classification using Deep Learning

### Overview
This project aims to classify images of different types of footwear, specifically boots, sandals, and shoes, using a convolutional neural network (CNN). The model learns to differentiate between these categories based on features within the images, allowing accurate predictions on new data.

### Project Description
This deep learning project uses image data to train a model capable of recognizing different footwear types. Using a dataset of images organized into folders by category (Boot, Sandal, Shoe), the project applies a CNN architecture to achieve high accuracy in image classification. This model could be used in e-commerce applications or inventory management systems to automate image categorization of footwear.

### Key Features
- **Automated Image Classification**: Classifies images of footwear into predefined categories.
- **High Accuracy**: Achieves competitive performance using a CNN model with several convolutional layers.
- **Customizable Architecture**: Model can be extended with additional layers or modified for better performance.
- **Preprocessing and Data Augmentation**: Includes techniques to enhance the robustness and generalization of the model.

### Technologies Used
- **Python**: Primary programming language.
- **TensorFlow & Keras**: Libraries for building and training the deep learning model.
- **NumPy & Pandas**: For data manipulation and analysis.
- **Matplotlib**: Used for visualizing training results.
- **ImageDataGenerator**: Employed for image preprocessing and augmentation.

### Project Structure
```
.
├── data/
│   ├── Boot/
│   ├── Sandal/
│   ├── Shoe/
├── model/
│   ├── model.py              # Model architecture and training script
│   ├── evaluate.py           # Evaluation script for testing accuracy
├── notebooks/
│   ├── EDA.ipynb             # Jupyter Notebook for exploratory data analysis
├── README.md                 # Project documentation


### How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/footwear-classification.git
   cd footwear-classification

   ```

2. **Data Preparation**:
   Ensure that the `data` directory contains the folders `Boot`, `Sandal`, and `Shoe`, each populated with the relevant images.

3. **Train the Model**:
   Run the main script to start the training process:
   ```bash
   python main.py
   ```

5. **Evaluate the Model**:
   After training, you can evaluate the model on test data using:
   ```bash
   python model/evaluate.py
   ```

### Results
- **Training Accuracy**: 90.62%
- **Validation Accuracy**: 75.00%
- **Training Loss**: 0.3065
- **Validation Loss**: 0.5651

The model demonstrates a high accuracy on training data and a reasonable accuracy on validation data, indicating it has learned relevant patterns. Additional adjustments to reduce overfitting are recommended for further improvements.

### Future Improvement
- **Hyperparameter Tuning**: Experiment with learning rate adjustments, optimizer selection, and batch sizes for better performance.
- **Data Augmentation**: Additional augmentation techniques could further improve generalization.
- **Model Architecture**: Testing different architectures or deeper models, such as ResNet or Inception, may enhance classification accuracy.
- **Transfer Learning**: Utilizing pre-trained models like VGG16 or MobileNet may improve results with fewer labeled images.
  
--- 
