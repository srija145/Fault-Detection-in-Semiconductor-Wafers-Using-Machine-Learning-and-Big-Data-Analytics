### Fault Detection in Semiconductor Wafers using ML Models.
 
**Project Overview:**
This project aims to address the issue of predicting the quality of wafers in semiconductor manufacturing using machine learning techniques. The dataset consists of various sensor readings from the wafer manufacturing process, and the objective is to predict whether a wafer is "good" or "bad" based on these sensor readings.
 
### Dataset:
We obtain values from 590 sensors for every wafer, and using those numbers, we determine if the wafer is in good working order (1) or not (0).

**Dataset: https://drive.google.com/drive/folders/1ogyJ0yUBqcmK7FtqSmVVNyiNaAtXj2ab?usp=drive_link**
 
We have developed all the code in google Collab and all the necessary libraries installation is mentioned in the first cell.

We have mounted the google drive and connected to the dataset in the “Data_603_Project” folder (Public access is given) which is present in the above-mentioned dataset link.
1. All the Box plots are saved in “BoxPlots” folder.
2. All the histograms are saved in “Histograms” folder.
3. Google Collab ipynb notebook is saved in “Code” folder.
4. All the training data is present in “Training_Data” folder. 

### Here is the parent folder “Data_603_Project” link: 
https://drive.google.com/drive/folders/10j3MC2IWC6yzPTtHi-K4tE6-jNgRminP?usp=drive_link

Installation:
!pip3 install numpy 
!pip3 install pandas 
!pip3 install pyspark
!pip3 install kneed 
!pip3 install scikit-learn
!pip3 install matplotlib
!pip3 install seaborn
!pip3 install pydrive 
!pip3 install gdown
 
### We have followed the below steps.

### Data Preprocessing:
In this step, we perform different steps of data validation and preprocessing like
●	Null values check
●	Outliers check
●	Standard deviation check
●	Target Class Imbalance check
●	Data Distribution check
 
### Null values check: 

In this step, we have checked the null values in each column and used the mean value to update to null values as all the sensor data is numeric data.

### Outliers check:

In this step, we checked for the outliers using box plots and identified some outliers so, we did used Winsorization technique to handle the outliers.

### Standard Deviation check:

In this step, we checked the standard deviation of all the columns and identified around 116 columns with 0 standard deviation. As this means that all these sensors’ data is same for all the wafers, there is meaningful insights from these columns so, we decide to drop these 116 columns.

 ### Target Class Imbalance check:

In this step, we checked the target class and found the data is in 70:30 ratio. So, we used the SMOTE technique to balance the target class.

### Data Distribution Check: 

In this step, we checked the distribution and scale of each sensor data and identified that the scale is different for different sensors. So, we have used standardization technique to bring the data distribution to normal distribution and put all the columns in the same scale.

### Model Training:
**Clustering:**
As we have 590 columns and huge number of data, we decide to go with semi supervised learning method to bring the best predictive model. In this process, we initially did K-means clustering and using elbow plot we have grouped the data into 4 clusters.

**Classifier Model:** 
For each cluster, we have trained three models namely logistic regression, **Decision Tree** and **Random Forest Classifier** and evaluated the results at each cluster. Out of which for all the clusters **Random Forest classifier gave good accuracy**.


