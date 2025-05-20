#  Supervised Contrastive Learning & Transfer Learning Across Modalities

This repository contains all deliverables for the assignment exploring modern supervised contrastive loss, transfer learning across modalities (image, video, audio, NLP), zero-shot learning, and advanced image classification techniques using state-of-the-art models like EfficientNet, BiT, MLP-Mixer, and ConvNeXt V2.

---

The work is combined in 4 colabs and 1 video 

## ✅ Part 1: Supervised Contrastive Learning vs. Softmax

### 📘 Colab Notebooks:
- [`Supervised_Contrastive_vs_Softmax.ipynb`](https://colab.research.google.com/drive/1Zu-mpS9AA7_ZTcqQJqzrjTuXxqc0X7W7?usp=sharing): Implements both traditional softmax-based classification and supervised contrastive learning on a common dataset (e.g., CIFAR-10 or Fashion-MNIST), comparing performance and visualizing feature embeddings.

### 📹 Video Walkthrough:
- [Video Walkthrough: Supervised Contrastive Learning](https://youtu.be/5F1Kb3r6uYQ)

---

## ✅ Part 2: Transfer Learning on Various Modalities

### 🖼️ Image Transfer Learning:
- [`Image_TransferLearning_FeatureExtractor_FineTuning.ipynb`](https://colab.research.google.com/drive/1EXiO3oi4g51n9Fd8GXHB4zg-y1mSlaIg?usp=sharing): Transfer learning on Cats vs Dogs dataset using EfficientNet as a feature extractor and fine-tuned model.

### 🔊 Audio Transfer Learning:
- [`Audio_TransferLearning_YAMNet.ipynb`](https://colab.research.google.com/drive/1EXiO3oi4g51n9Fd8GXHB4zg-y1mSlaIg?usp=sharing): Implements YAMNet for audio classification with transfer learning.

### 🎞️ Video Transfer Learning:
- [`Video_TransferLearning_ActionRecognition.ipynb`](https://colab.research.google.com/drive/1EXiO3oi4g51n9Fd8GXHB4zg-y1mSlaIg?usp=sharing): Uses TF Hub to perform video action recognition with I3D architecture.

### 📝 NLP Transfer Learning:
- [`NLP_TransferLearning_TFHub_TextClassification.ipynb`](https://colab.research.google.com/drive/1EXiO3oi4g51n9Fd8GXHB4zg-y1mSlaIg?usp=sharing): Uses pre-trained BERT for text classification using TensorFlow Hub.

### 📹 Video Walkthrough:
- [Video Walkthrough: Transfer Learning Modalities](https://youtu.be/5F1Kb3r6uYQ)

---

## ✅ Part 3: Zero-Shot and State-of-the-Art Transfer Learning

### 🖼️ CLIP Zero-Shot Learning:
- [`CLIP_ZeroShot_ImageClassification.ipynb`](https://colab.research.google.com/drive/16j1Qy66wUuJV_5TpcZ_KzemtO8DgpW1L?usp=sharing): Demonstrates OpenAI’s CLIP for zero-shot image classification.

### 🌼 BigTransfer on Flowers:
- [`BigTransfer_TFHub_TransferLearning.ipynb`](https://colab.research.google.com/drive/16j1Qy66wUuJV_5TpcZ_KzemtO8DgpW1L?usp=sharing): Demonstrates transfer learning using BigTransfer (BiT) model on the Flowers dataset.

### 📹 Video Walkthrough:
- [Video Walkthrough: Zero-Shot & BiT](https://youtu.be/5F1Kb3r6uYQ)

---

## ✅ Part 4: Vision Classifiers on Benchmark Datasets

### 📊 Datasets: MNIST, Fashion MNIST, CIFAR-10

#### Each Dataset Contains:
1. EfficientNet Transfer Learning
2. BiT Transfer Learning
3. MLP-Mixer
4. ConvNeXt V2

### 🧪 Colab Notebooks:
- [`MNIST_Vision_Models.ipynb`](https://colab.research.google.com/drive/112oWSoVnXL3FWGnqe8oCC_azC9NICi16?usp=sharing)
- [`FashionMNIST_Vision_Models.ipynb`](https://colab.research.google.com/drive/112oWSoVnXL3FWGnqe8oCC_azC9NICi16?usp=sharing)
- [`CIFAR10_Vision_Models.ipynb`](https://colab.research.google.com/drive/112oWSoVnXL3FWGnqe8oCC_azC9NICi16?usp=sharing)

### 📹 Video Walkthrough:
- [Video Walkthrough: Benchmark Dataset Models](https://youtu.be/5F1Kb3r6uYQ)

---

## ✅ Part 5: Medical Image Classification

### 🩻 X-ray Pneumonia Classification:
- [`Xray_Pneumonia_ConvNet.ipynb`](https://colab.research.google.com/drive/112oWSoVnXL3FWGnqe8oCC_azC9NICi16?usp=sharing): Implements ConvNet and EfficientNet for pneumonia detection from chest X-rays.

### 🧠 3D CT Scan Classification:
- [`3D_CTScan_Classification.ipynb`](https://colab.research.google.com/drive/112oWSoVnXL3FWGnqe8oCC_azC9NICi16?usp=sharing): Uses 3D convolutional networks to classify CT scan volumes.

### 📹 Video Walkthrough:
- [Video Walkthrough: Medical Imaging](https://youtu.be/5F1Kb3r6uYQ)
