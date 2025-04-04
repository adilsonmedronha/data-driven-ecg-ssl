# Abstract
------------------------------------------------
Affect recognition aims to detect a person's affective state based on observables, with the goal to e.g. improve human-computer interaction. Long-term stress is known to have severe implications on wellbeing, which call for continuous and automated stress monitoring systems. However, the affective computing community lacks commonly used standard datasets for wearable stress detection which a) provide multimodal high-quality data, and b) include multiple affective states. Therefore, we introduce WESAD, a new publicly available dataset for wearable stress and affect detection. This multimodal dataset features physiological and motion data, recorded from both a wrist- and a chest-worn device, of 15 subjects during a lab study. The following sensor modalities are included: blood volume pulse, electrocardiogram, electrodermal activity, electromyogram, respiration, body temperature, and three-axis acceleration. Moreover, the dataset bridges the gap between previous lab studies on stress and emotions, by containing three different affective states (neutral, stress, amusement) [1].


# Subjects
------------------------------------------------
17 subjects participated in the study. However, due to sensor malfunction, data of two subjects (S1
and S12) had to be discarded. For this study the Subject 17 is used as **test data** while the remain 
subjects is used as training data using cross-validation. 


# Annotations
------------------------------------------------
Annotations/meanings in these files are:

1 = baseline, 2 = stress, 3 = amusement, 4 = meditation

# Channels & Signals
------------------------------------------------
In this study only the chest data acquired with RespiBAN is considered.The signals was segmented into 2 seconds segments and sampling rate was reduced to 360 Hz. Each channel is considered a isolated univariate time series*. Due the final data volume, a subset of segments was selected. In order to maintain the original data distribution was used a uniform distribution stratified by subject and annotation (classes).

Considered channels:
* ECG - Electrocardiogram (mV)
* EDA - Electrodermal activity (μS)
* EMG - Electromyogram (mV)
* RESP - Respiration (%)
* TEMP - Body temperature (°C)


> *Obs: Accelerometer was discarded once its a non-physiological signal.


# References
------------------------------------------------
[1] Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven. 2018.
Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. In 2018
International Conference on Multimodal Interaction (ICMI ’18), October 16–20, 2018, Boulder, CO,
USA. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3242969.3242985