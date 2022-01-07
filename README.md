# XAI_ECG
Code to generate ECG figures from MIT BIH arrhythmia database, train CNN models (RESNET-50) on those images and produce explainability using 3 different methods.

Steps to properly run the code:
  1. Change the config.json file according to your needs (change the path to the dataset and configurations of the resnet-50);
  2. Run ecg_classifier.py to train the models - writes in config.json where the model is saved;
  3. To test the model, use the notebooks/testing.ipynb;
  4. To make the attribution maps, use the appropriate notebooks (according to their names).
