# Wavelet Optimization for ECG Signal Classification
Generating the wavelet that best represents the ECG beats in terms of discrimination capability 

## Overview

This projectpresents a novel method to enhance the classification accuracy of ECG signals by
optimizing wavelet design specifically for this purpose. While traditional wavelets, such
as Daubechies and Symlet, are widely used in ECG signal processing, the researchers
believe that they do not provide the highest accuracy in classification tasks. Through this
paper they addresses this limitation by proposing a new approach that integrates wavelet
design with the classification process to achieve a better results.

The proposed method utilizes the polyphase representation of wavelet filter banks, enabling wavelet to design through a set of angular parameters. These parameters are optimized using a Particle Swarm Optimization (PSO) algorithm to maximize the accuracy of
ECG beat classification. The classification process uses a Support Vector Machine(SVM).
The study incorporates both morphological and temporal features extracted from ECG
signals, for the classification.
The study validate their method using the MIT-BIH Arrhythmia Database, focusing
on a subset of ECG recordings and classifying beats into six categories
N (·) (Normal beat),
L (Left bundle branch block beat),
R (Right bundle branch block beat),
A (Atrial premature beat),
V (Premature ventricular contraction),
/ (Paced beat),

## Requirements

Make sure you have Python installed along with the necessary libraries. 

## Functionality
Data Loading: Load ECG signal data from the MIT-BIH database.
Data Preprocessing: Split the dataset into training and testing sets.
Create custom wavelets using sherlock Monro Algorithm 
Wavelet Transformation: Apply wavelet decomposition to the ECG signals.
Optimize the parameters of wavelet transformation using Particle Swarm Optimization (PSO).
Classification: Evaluate the classification performance using k-fold cross validation using SVM.
![image](https://github.com/user-attachments/assets/57a21939-e1ec-45b9-a142-5ae13b4ec2b1)

## Results
After 20 iterations a generated mother wavelet is displayed below.
![mother wavelet for best accuracy](https://github.com/user-attachments/assets/01893bd9-da62-40da-9d39-19a5897f62b1)

## References
Abdelhamid Daamouche, Latifa Hamami, Naif Alajlan, Farid Melgani,
A wavelet optimization approach for ECG signal classification,
Biomedical Signal Processing and Control, 
Volume 7, Issue 4,
2012,
Pages 342-349,
ISSN 1746-8094,
https://doi.org/10.1016/j.bspc.2011.07.001.

(https://www.sciencedirect.com/science/article/pii/S1746809411000772)

Abstract: Wavelets have proved particularly effective for extracting discriminative features in ECG signal classification. In this paper, we show that wavelet performances in terms of classification accuracy can be pushed further by customizing them for the considered classification task. A novel approach for generating the wavelet that best represents the ECG beats in terms of discrimination capability is proposed. It makes use of the polyphase representation of the wavelet filter bank and formulates the design problem within a particle swarm optimization (PSO) framework. Experimental results conducted on the benchmark MIT/BIH arrhythmia database with the state-of-the-art support vector machine (SVM) classifier confirm the superiority in terms of classification accuracy and stability of the proposed method over standard wavelets (i.e., Daubechies and Symlet wavelets).

Keywords: Classification; Discrete wavelet transform (DWT); Electrocardiogram (ECG) signals; Particle swarm optimization (PSO); Support vector machines (SVM)
