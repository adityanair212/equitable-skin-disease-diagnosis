Skin Disease Diagnosis using an Ensemble of Deep Learning Models
About This Project
This project presents a robust and equitable system for the diagnosis of various skin diseases from dermatoscopic images. It leverages an ensemble of multiple deep learning models to achieve higher accuracy and more reliable predictions than a single model could. The system is capable of classifying skin lesions into several categories, such as Basal Cell Carcinoma, Eczema, and Melanocytic Nevi, with a strong focus on fair performance across diverse skin tones.

To provide insight into the models' decision-making processes, this project integrates Grad-CAM (Gradient-weighted Class Activation Mapping), which generates visual heatmaps highlighting the regions of an image that are most important for a given prediction.

The project also includes a user-friendly web application built with Flask, allowing users to upload an image of a skin lesion and receive a diagnosis from the trained ensemble model in real-time.

Addressing Algorithmic Bias in Dermatology
A significant challenge in medical AI is algorithmic bias, where models trained on unrepresentative data perform poorly on minority populations. In dermatology, datasets are often heavily skewed towards lighter skin tones, leading to AI systems that are less accurate for individuals with darker skin. This can worsen existing health disparities.

This project directly confronts this issue by:

Creating a Composite Dataset: We merged three public datasets (PAD-UFES-20, Stanford AIMI, and DermNet) to build a more diverse and representative collection of images across various skin tones.

Using Data Augmentation: Techniques such as contrast and brightness adjustments were used to simulate different lighting conditions and further enhance the model's ability to generalize across all skin types.

Our goal is to create a fair and equitable diagnostic tool that is reliable for all populations.

Key Features
Multi-Model Ensemble: Combines predictions from DenseNet-121, EfficientNet, and MobileNetV3 for enhanced classification performance.

Fairness and Equity: Specifically designed and trained on a diverse dataset to reduce skin tone bias and ensure equitable performance.

Model Interpretability: Uses Grad-CAM to generate heatmaps, making the "black box" of deep learning more transparent.

Confidence-Based Rejection: Includes a mechanism to flag low-confidence predictions as "Uncertain," advising referral to a specialist and preventing overconfident errors.

Web Application: An intuitive web interface for easy diagnosis by uploading an image.

Models Used
The ensemble is composed of the following pre-trained and fine-tuned models:

DenseNet-121

EfficientNet

MobileNetV3

Project Structure
The project is organized into a clean, modular structure to ensure clarity and ease of maintenance:

skin-lesion-project/
│
├── app/              # All files related to the Flask web application
├── data/             # (Ignored by Git) Raw and processed image datasets
├── models/           # (Ignored by Git) Trained model weights (.pth files)
├── reports/          # Final figures, confusion matrices, and result summaries
├── src/              # Core Python source code for the project
│   ├── data_preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── visualization.py
│
├── .gitignore        # Specifies which files/folders to ignore in Git
├── README.md         # This file
└── requirements.txt  # List of Python dependencies

Setup and Installation
Follow these steps to set up the project on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

1. Clone the Repository
Clone this repository to your local machine:

git clone https://github.com/your-username/skin-lesion-project.git
cd skin-lesion-project

2. Install Dependencies
Install all the required Python libraries using the requirements.txt file. It's highly recommended to use a virtual environment.

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

3. Download Data and Models
Due to their large size, the dataset and trained models are not tracked by Git.

Download the dataset: [<< INSERT YOUR DATASET DOWNLOAD LINK HERE >>]

Unzip the dataset and place the processed_dataset folder inside the data/ directory.

Download the trained models: [<< INSERT YOUR MODELS DOWNLOAD LINK HERE >>]

Place the downloaded .pth files into the models/ directory.

How to Use the Project
You can run the project in several ways: training a new model, evaluating an existing one, or launching the web application.

Training a New Model
To train one of the models from scratch, you can run the training script from the src/ directory.

python src/training.py --model densenet  # Example for training DenseNet

Evaluating a Model
To evaluate the performance of a trained model on the test set, use the evaluation script.

python src/evaluation.py --model efficientnet

Launching the Web Application
To start the Flask web application for interactive diagnosis:

python app/app.py

Once the server is running, open your web browser and navigate to http://127.0.0.1:5000.

Contributing
Contributions are welcome! If you have suggestions for improvements or want to add new features, please feel free to fork the repository, make your changes, and submit a pull request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

License
This project is distributed under the MIT License. See LICENSE for more information.

Acknowledgements
This project uses a composite dataset from PAD-UFES-20, Stanford AIMI, and DermNet.

Special thanks to the creators of the PyTorch, Flask, and Grad-CAM libraries.