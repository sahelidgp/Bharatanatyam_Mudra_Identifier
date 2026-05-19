# Bharatnatyam Mudra Recognizer

This project is a web-based application built with Streamlit that uses computer vision to identify Bharatnatyam hand gestures (mudras). It can analyze mudras from uploaded images, videos, and a live webcam feed in real-time.

The core of the recognition system is a custom-trained neural network that classifies hand landmarks extracted by Google's MediaPipe framework, providing an interactive and educational tool for dancers and enthusiasts.



## Features

+ **Mudra Library**: A comprehensive gallery of 50 different mudras, each with a reference image and a detailed description.
+ **Image Recognition**: Upload a static image (JPG, PNG) to detect and identify the mudra being performed.
+ **Video Analysis**: Upload a video file (MP4, MOV) to get a summary and timeline of all mudras performed throughout the video.
+ **Webcam Snapshot**: Use your webcam to take a single picture and get an instant classification.
+ **Live Real-Time Feed**: Activate your webcam for a live video stream where the application identifies mudras as you perform them.

-----

## Technology Stack

+ **Python**: Core programming language.
+ **Streamlit**: For creating the interactive web application UI.
+ **TensorFlow / Keras**: For building and training the mudra classification model.
+ **MediaPipe**: For high-fidelity hand and finger tracking to extract landmarks.
+ **OpenCV**: For image and video processing.
+ **Scikit-learn**: For data preprocessing (scaling and label encoding).

-----

## Project Structure

```
bharatanatyam-mudra-recognizer/
│
├── .vscode/
│   └── settings.json           # (Recommended) VS Code workspace settings
├── model_artifacts/            # Stores the trained model files (created after training)
├── mudra_images/               # Reference images for the Mudra Library
├── Bharatanatyam-Mudra-Dataset/ # The training image dataset
├── venv/                       # Virtual environment directory
│
├── app.py                      # The main Streamlit application file
├── model_train.py              # Script to train the classification model
├── requirements.txt            # List of all Python dependencies
└── README.md                   # This documentation file
```

-----

## Getting Started: Setup and Installation

Follow these steps carefully to set up the project environment and run the application on your local machine.

### Step 1: Prerequisites

+ [Python](https://www.python.org/downloads/) (version 3.10 - 3.11 is recommended).
+ [Git](https://git-scm.com/downloads/) for cloning the repositories.

### Step 2: Clone the Required Repositories

First, clone this project repository. Then, navigate into the new directory and clone the image dataset repository which is required for training.

```bash
# Clone the main application repository
git clone https://github.com/your-username/bharatanatyam-mudra-recognizer.git

# Navigate into the project directory
cd bharatanatyam-mudra-recognizer

# Clone the image dataset inside the project directory
git clone https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset.git
```

### Step 3: Set Up the Python Virtual Environment

Using a virtual environment is crucial for managing project-specific dependencies and avoiding conflicts with other projects.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
.\\venv\\Scripts\\activate

# On macOS/Linux:
source venv/bin/activate
```

After activation, your terminal prompt will be prefixed with `(venv)`.

### Step 4: Install All Dependencies

Install all the necessary Python libraries at once using the `requirements.txt` file.

```bash
# Ensure your virtual environment is active before running this
pip install -r requirements.txt
```

### Step 5: Configure VS Code Interpreter (Recommended)

To ensure your code editor (like Visual Studio Code) recognizes the installed libraries, you must select the correct Python interpreter.

1.  Open the project folder in VS Code.
2.  Press **`Ctrl + Shift + P`** to open the Command Palette.
3.  Type **`Python: Select Interpreter`** and select it from the list.
4.  Choose the interpreter that includes **`(venv)`** in its name and path. It will look like `Python 3.x.x ('venv') ./venv/Scripts/python.exe` and is usually marked as **"Recommended"**.

> **Troubleshooting**: If you still see import errors (red squiggly lines), simply reload the VS Code window by running **`Developer: Reload Window`** from the Command Palette.

### Step 6: Prepare the Mudra Library Images

The web app's Mudra Library requires a set of reference images. This is a one-time manual setup step.

1.  Create a new folder named `mudra_images` in the project's root directory.
2.  For each mudra sub-folder inside `Bharatanatyam-Mudra-Dataset` (e.g., `Alapadmam`), copy **one** representative image.
3.  Paste that image into your `mudra_images` folder and rename it to match the mudra's name exactly (e.g., `Alapadmam.jpg`, `Anjali.jpg`).
4.  Repeat this process for all 50 mudras.

-----

## Running the Application

The project is run in two phases: a one-time model training phase, followed by running the web application.

### Phase 1: Train the Model (One-Time Step)

This script processes the dataset, trains the neural network, and saves the required model files.

Make sure your virtual environment is activated (`(venv)` is visible in your terminal), then run:

```bash
python model_train.py --dataset_dir ./Bharatanatyam-Mudra-Dataset --output_dir ./model_artifacts
```

This will create the `model_artifacts` folder containing `model.h5`, `scaler.pkl`, and `label_encoder.pkl`.

### Phase 2: Launch the Web App

Once the model is trained and the artifacts are in place, you can launch the interactive web application.

```bash
streamlit run app.py
```

Your web browser will automatically open a new tab with the application running. You can now explore all its features\!

-----

## Acknowledgments

  - The image dataset used for training this model was provided by Jisha R. and is available at the [Bharatanatyam-Mudra-Dataset GitHub repository](https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset).