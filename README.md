\documentclass{article}
# Adversarial Protection in Face Forgery Detection
  This project reproduces the backdoor attacks on models trained to detect face forgeries, examining how these attacks impact the model's ability to distinguish between real and fake faces.
  The project also explores defense mechanism to identify and neutralize these attacks without comprimising the model's performance<br>

  [link to the project report](https://drive.google.com/file/d/19dICnPXcaYSrgrU3HtvcrESikHkwM5ko/view?usp=drive_link)
  
# Dataset preparation
we use three datasets from kaggle:
1. **Fakeforensics++** and **celebDF(V2)**: These have both real and fake faces made with different deepfake techniques.
2. **Real-v/s-Fake**: This has real and fake faces made with styleGAN
here are the links:<br>
 i)[Faceforensics++](https://drive.google.com/file/d/1KDMFUdNPZ1fVKcZMhh0OJ0939rHnlv00/view?usp=drive_link)<br>
 ii)[CelebDF(v2)](https://drive.google.com/file/d/1bmBvCR3R4h_aIpisXOQpy-MqJKN_M_Uk/view?usp=drive_link)<br>
 iii)[Real-v/s-Fake](https://drive.google.com/file/d/1eqNqWSSVk3eHjvZqXYsVk_fDTgwKQfkr/view?usp=drive_link)<br>
 The datasets links uploaded here are preprocessed.In general, to extract facial regions from video frames in the dataset, a pre-trained face landmark detection model is utilized. The implementation of this process is available in the file \texttt{src/preprocess/crop\_dlib.py}.

Download the pre-trained face landmark detection model [here](https://github.com/VamshiNarmety/Adversarial-Protection-in-Face-Forgery-Detection/blob/main/src/preprocess/shape_predictor_81_face_landmarks.dat)<br>


# Note
this is just the tentative one.The complete project repository will be updated soon.
