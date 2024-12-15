# Adversarial Protection in Face Forgery Detection
  This project reproduces the backdoor attacks on models trained to detect face forgeries, examining how these attacks impact the model's ability to distinguish between real and fake faces.
  The project also explores defense mechanism to identify and neutralize these attacks without comprimising the model's performance<br>
  here are the links to selected papers:<br>
   i) [Poisoned Forgery Face: Towards Backdoor Attacks on Face Forgery Detection](https://arxiv.org/abs/2402.11473v1)<br>
   ii)[BadActs: A Universal Backdoor Defense in the Activation Space](https://arxiv.org/abs/2405.11227v1)
# Datasets
we use three datasets from kaggle:
1. **Fakeforensics++** and **celebDF(V2)**: These have both real and fake faces made with different deepfake techniques.
2. **Real-v/s-Fake**: This has real and fake faces made with styleGAN
here are the links:<br>
 i)[Faceforensics++](https://www.kaggle.com/datasets/farhansharukhhasan/faceforensics1600-videospreprocess/data)<br>
 ii)[CelebDF(v2)](https://www.kaggle.com/datasets/shivendrasinha/celeb-dfv2-processed/data)<br>
 iii)[Real-v/s-Fake](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)<br>
The CelebDF(v2) datasets has video frames, to extract only the facial regions we use a pre-trained face landmark detection, the code is available in this file (src/preprocess/crop_dlib.py)<br>


# Note
this is just the tentative one.The complete project repository will be updated soon.
