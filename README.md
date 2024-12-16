# Adversarial Protection in Face Forgery Detection
  This project reproduces the backdoor attacks on models trained to detect face forgeries, examining how these attacks impact the model's ability to distinguish between real and fake faces.
  The project also explores defense mechanism to identify and neutralize these attacks without comprimising the model's performance<br>

  [link to the project report](https://drive.google.com/file/d/19dICnPXcaYSrgrU3HtvcrESikHkwM5ko/view?usp=drive_link)

# setup the project
The code was run on Ubuntu with the following setup:

1. Install Python 3.10.x

2. From the terminal, run the following:

   ```bash
   pip3 install -r requirements.txt
and also code was run on kaggle notebooks for accessing GPUs, so make sure to change the paths for using any file in the code depending on whether you are running the code on your local machine or any cloud services like google colab/kaggle.

# Dataset preparation
we use three datasets from kaggle:
1. **Fakeforensics++** and **celebDF(V2)**: These have both real and fake faces made with different deepfake techniques.
2. **Real-v/s-Fake**: This has real and fake faces made with styleGAN
here are the links:<br>
 i)[Faceforensics++](https://drive.google.com/file/d/1KDMFUdNPZ1fVKcZMhh0OJ0939rHnlv00/view?usp=drive_link)<br>
 ii)[CelebDF(v2)](https://drive.google.com/file/d/1bmBvCR3R4h_aIpisXOQpy-MqJKN_M_Uk/view?usp=drive_link)<br>
 iii)[Real-v/s-Fake](https://drive.google.com/file/d/1eqNqWSSVk3eHjvZqXYsVk_fDTgwKQfkr/view?usp=drive_link)<br>
 The datasets links uploaded here are preprocessed.In general, to extract facial regions from video frames in the dataset, a pre-trained face landmark detection model is utilized. The implementation of this process is available in the file `src/preprocess/crop_dlib.py`.


Download the pre-trained face landmark detection model [here](https://github.com/VamshiNarmety/Adversarial-Protection-in-Face-Forgery-Detection/blob/main/src/preprocess/shape_predictor_81_face_landmarks.dat)<br>

# Test model on clean Datasets
After setting up the dataset, you can train and test the model on clean,unmodified datasets to ensure it works well on clean data.<br>
**Train the model:** To train the model on clean datasets, run the `train_test_model(cleandata).py` file.
- The purpose of testing on clean data is explicitly mentioned to ensure the model works well before proceeding to the attacks.
- 
# Note
this is just the tentative one.The complete project repository will be updated soon.
