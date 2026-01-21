# DermAI (Derma AI Images) — Skin Condition Classifier

## About the Tool; My Thoughts

This project is based on a previous work experience in the summer of 2023. During this experience, I worked for the Aga Khan Development Network's Digital Health Resource Centre, where I gathered a dataset to be used for an AI/ML identification project (this project was already in development long before I joined the team and is an ongoing effort for the team).

However, the overall project the team was working towards was beyond the scope of my internship. It was at the beginning of the next year (in 2024) that I wanted to challenge myself: after being interested in AI and looking into classification tools, could I program one myself, using the data I collected? This project serves as a practical use case of what I've learned, and while it's still **currently a work in progress**, I have made significant headway with the project.

While I did spend most of 2024 immersed in other opportunities, ranging from learning about hardware and software systems in academia to starting a series of internships in data analysis and software development, it was there that I got myself more familiar with tools that can be used to analyze datasets and work with them, like Tensorflow, TFRecord, Python and many more technologies. 

The project, in its current state, reflects my knowledge set and ability to apply it. This resulted in a working prototype that was able to analyze and detect with an accuracy of around 80%, but I think I can do better. This is why what's currently uploaded is an incomplete state of the project that I'm working on locally. But I hope to revamp things in this repo very soon and present a project that is complete in function.

In the meantime, here's a quick TLDR breakdown with more info about the project, the steps I've taken and what I hope to accomplish with it:  

---

## Information about the tool
What it is: A personal ML project to train an image classifier on dermatology condition photos, with a clean data pipeline, leakage-safe splits, imbalance-aware training, and a planned mobile demo deployment.

> **Disclaimer:** This is an educational/research prototype and is **not** a medical diagnostic tool.

---


DermAI is a personal machine learning project focused on classifying common skin conditions from images.  
The goal of the project is to build a **clean, leakage-safe training pipeline**, train a CNN on an imbalanced dermatology dataset, and eventually deploy the model in a simple desktop and mobile demo.

> This project is for learning and experimentation only and is **not a medical diagnostic tool**.

---

## Project Overview

- Dataset: dermatology images grouped by **condition** and **age group**
- Frameworks: Python, TensorFlow / Keras
- Model: CNN with EfficientNet backbone
- Focus areas:
  - Data hygiene (stratified splits, leakage detection)
  - Handling class imbalance
  - Reproducible training pipeline
  - Practical deployment (planned)

---

## Current Progress

### Dataset indexing
- Each image entry includes:
  - file path  
  - condition label  
  - age group  
  - data source  
- Total images indexed: **1993**

---

### Train/validation/test split
- Used stratified splitting based on **condition × age group** to preserve distributions.
- Final split sizes:
  - Train: 1593  
  - Validation: 199  
  - Test: 200  
- Verified that no images appear in more than one split.

---

### Duplicate and leakage handling
- Checked for data leakage using:
  - SHA1 hashes (exact duplicates)
  - dHash (visually identical or near-identical images)
- Initially, I found many duplicates crossing splits.
- Fixed this by moving duplicate groups entirely into the training set.
- After cleanup:
  - No exact duplicates cross splits
  - No identical perceptual duplicates cross splits

---

### TFRecords and training (currently being remodelled for greater accuracy)


---

Thanks for reading and checking this project out!
