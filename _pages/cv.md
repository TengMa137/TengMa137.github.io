---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* M.S. in Autonomous System, UNITN,IT / KTH,SE, 2020-2022
* B.S. in Automation, Jilin University, CN, 2015-2019

Area of Speciality
======
Robot Perception and State Estimation, Computer Vision, Machine Learning

Tech Stack
======
* C/C++, Python, MATLAB
* Docker, Git
  
Work experience
======

LiDAR 4D Panoptic Segmentation
------
* Internship in the department of emerging technology of Volvo CE *
* Built point cloud dataset for Volvo CE, including data collection, data denoising and data annotation.
* Adopted the state-of-the-art model 4D-PLS to achieve 4D panoptic segmentation of point cloud data, the goal of which is to obtain the semantic meaning of each point and the ability to track moving objects in a sequence of point clouds simultaneously.
* Optimized hyperparameters of the deep learning model using Bayesian Optimization with TPE surrogate model and improved the main model evaluation metric LSTQ from 93% to 94%.
>_Tools: Git, Pytorch, Python, Linux_

Projects
======

Semantic Visual-inertia Odometry
------
* Combined YOLOv8 and orb-slam3 to obtain semantic instance-level motion information.
* Estimated the state of objects with a focus on relative and absolute velocity.
>_Tools: Opencv, C++_ 

Local Chatbot
------
* Built a web UI that run locally for a chatbot powered by llamacpp model using Streamlit.
* Supported chat with humam, PDF file, Website page and more (continuously updating).
* A chatgpt-like experience but with all the data stored locally and model running on a PC in a reasonable inference speed.
>_Tools: Langchain, Streamlit, Python_ 

GPS/IMU Localization
------
* Investigated the application of Kalman Filter in fusing IMU with other sensors.
* Implemented error state Kalman Filter in Python to fuse IMU and GPS data. Sensor data was from the first sequence of odometry in Kitti dataset.
>_Tools: Python, Matlab_ 
  
Pedestrian Attribute Recognition & ReID
------
* Identified the individual in the query image set who was previously displayed in the training set by recognizing the attributes such as age, gender, color of wearing etc. Images and annotations were from Market-1501 dataset.
* Modify a pretrained Resnet50 in Pytorch as the backbone, processed the annotation data and output the results in csv format using Pandas.
>_Tools: Pytorch, Pandas, Python_ 

MBS Modeling and Simulation 
------
* Collaborated with teammates to build a model of a robot leg and conducted kinematic & dynamic analysis.
* Validated the numerical solutions of Lagrange equations from Maple RK4 solver in Matlab.
>_Tools: Maple, Matlab, Simulink_ 

Point Cloud Registration
------
* Studied different handcraft descriptors in feature extraction such as ISS, SIFT3D, FPFH.
* Estimated the transformation matrix of point clouds with RANSAC and ICP. The point cloud data was generated from Unity.
>_Tools: PCL, C++, Unity_ 
