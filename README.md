# VariationalAutoencodersForAnomalyDetection
This repository contains all the required code for the assignment 

### Introduction: 
Although VAEs were principally designed as generative models for image and text generation, we exploit these qualities to be leveraged in the anomaly detection domain. 

### Dataset: 
The [NAB](https://github.com/numenta/NAB) corpus conntains 58 timeseries data files designed to provide data for research in streaming anomaly detection. For this project we use three datasets from here, namely: machine_temperature_system_failure.csv, nyc_taxi.csv, and Twitter_volume_CRM.csv. The combined windowed labels are taken to check if the anomalies have been detected correctly. 

### Models:
Three models were used for this approach one is a non bayesian approach, the other two are Variational Autoencoders with different frameworks.  
- Isolation Forest: ...

- VAE using Dense Layers: ...

- VAE using Conv1d layers: ..


VAE.ipynb: We just tested our model for a non time series dataset (KDD Cup 1999 Dataset), where it worked pretty well. 