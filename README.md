# 程式交接
[TOC]

## 1. Dynamic Localization System 主程式

DyLo的主程式分為三段程式碼 : 
(1) Wireless_Hashing.ipynb 主要對應到論文中的 Wireless Phase。
(2) Image_Hashing.ipynb 主要對應到論文中的 Image Phase。
(3) Particle_Filter.ipynb 主要對應到論文中的 Particle Filter。


### (1) Wireless_Hashing.ipynb

#### -- Input : 

    train data : ./data/0318_92589_train/wireless_training_set.csv
    
1. 在 train data 的部分，wireless_training_set.csv 會記錄實驗環境中對每個 Beacon 收集到的 RSSI 值，後續轉換成 Hashing 後供 test data 比對。
    
![](https://i.imgur.com/MHkQf12.png)

    test data :　./Data/0318_92589_test/{walk}/{phone}/wireless_fingerprint_avg_10_7_beacon_rate_10.csv
    
2. 在 test data 的部分，不同 {walk} 中的每一隻 {phone} 都會記錄實驗環境中對每個 Beacon 收集到的 RSSI 值，後續轉換成 Hashing 後與 train data 比對。其中 label 為 Ground truth。

![](https://i.imgur.com/rP61QSK.png)


#### -- Output :

    ./wireless_reference_point/{methods}_{walk}_bc_{beacon_count}_br_{beacon_rate}_wireless_reference_point.csv

在 Wireless_Hashing Output 的部分，會將 Wireless Phase 的結果儲存在 {methods}_{walk}_bc_{beacon_count}_br_{beacon_rate}_wireless_reference_point.csv 中，做為後續 Image Phase 的輸入供其使用。

其中 : 
1. label 為 Ground truth。
2. wireless_reference_point 為 Wireless Phase 的結果，供 Wireless Filtering 做為判斷使用。
3. voter_wireless_reference_point 為 Wireless Phase 的結果，做為後續 Dynamic Weighting 的輸入供其使用

![](https://i.imgur.com/284z2tG.png)


### (2) Image_Hashing.ipynb

#### -- Input : 

    train data : ./Data/0318_92589_train_image/image_info.csv
    
1. 在 train data 的部分，image_info.csv 會記錄實驗環境中所辨識到的每個物件，紀錄其種類、位置、大小，後續轉換成 Hashing 後供 test data 比對。

![](https://i.imgur.com/Mk27Gih.png)

    
    test data :　./Data/0318_92589_test/{walk}/{phone}/image_info.csv
    
2. 在 test data 的部分，不同 {walk} 中的每一隻 {phone} 都會記錄實驗環境中所辨識到的每個物件，紀錄其種類、位置、大小，後續轉換成 Hashing 後與 train data 比對。其中 label 為 Ground truth。

![](https://i.imgur.com/4MMiaiq.png)


    wireless reference point : ./wireless_reference_point/{methods}_{walk}_bc_{beacon_count}_br_{beacon_rate}_wireless_reference_point.csv

3. 將 Wireless Phase 暫存的取出使用 : 
wireless_reference_point 供 Wireless Filtering 做為判斷使用。
voter_wireless_reference_point 做為 Dynamic Weighting 的輸入供其使用

![](https://i.imgur.com/284z2tG.png)

#### -- Output :

    ./image_to_pf/92589_itp_{methods}_{walk}_omr_{obj_missing_rate}_bc_{beacon_count}_br_{beacon_rate}.csv

在 Image_Hashing Output 的部分，會將動態整合 Wireless Phase & Image Phase 的 Dynamic Weighting 的結果儲存在 92589_itp_{methods}_{walk}_omr_{obj_missing_rate}_bc_{beacon_count}_br_{beacon_rate}.csv 中，做為後續 Particle Filter 的輸入供其使用。

![](https://i.imgur.com/zKO1xmL.png)


### (3) Particle_Filter.ipynb

#### -- Input : 

    ./image_to_pf/92589_itp_{methods}_{walk}_omr_{obj_missing_rate}_bc_{beacon_count}_br_{beacon_rate}.csv

將儲存在 Image_Hashing Output 的 Dynamic Weighting 的結果讀取出來做為 Particle_Filter.ipynb 的輸入，並完成 Particle Filter。

#### -- Output :

會將 Dynamic Localization System 的定位結果計算成以下三種數據
1. MDE
2. CDF
3. SDE


## 2. Performance Evaluation 的變因參數

92589_itp_{methods}_{walk}_omr_{obj_missing_rate}_bc_{beacon_count}_br_{beacon_rate}.csv'

將會針對{}內的參數做解釋(除了切換參數，尚須改變程式內容)

### (1) Experimental Scenarios

    walk_list (walk)

Stationary, Scripted Walk, Free Walk 三種 Experimental Scenarios 透過 walk_list 參數進行切換

### (2) 5.5 Effects of the Number of Wireless Devices and Cameras

    methods

不同數量組合的實驗由 methods 進行切換

### (3) 5.6 Effects of Blocked Object Problem

    obj_missing_rate

Blocked Object Problem 中的 R 由 obj_missing_rate 進行切換

1 為 R = 1
8 為 R = 0.8
6 為 R = 0.6

### (4) 5.7 Effects of Number of Beacons

    beacon_count

Number of Beacons 中 Beacons 的數量由 beacon_count 進行切換

7 為 7 Beacons
6 為 6 Beacons
5 為 5 Beacons
4 為 4 Beacons

### (5) 5.8 Effects of Device Heterogeneity

    beacon_rate

Number of Beacons 中的 Group G, G‘, G“ 由 beacon_rate 進行切換

1 為 Group G
2 為 Group G‘
3 為 Group G“

## 3. 前置作業參考程式

此部分繼承自110林廷瑋學長交接的 Code，可將手機接收到的 Raw Data 轉換成程式所需的 csv 檔案

(1) Wireless_hashing_train.ipynb
(2) Wireless_hashing_test.ipynb
(3) Image_hashing_train.ipynb
(4) Image_hashing_test.ipynb
(5) Object_detection.ipynb
(6) obj_detection_train.py
(7) obj_detection_test.py

## 4. 其他參考程式

此部分整理出實驗所用到的專案

### (1) 手機 App 

    https://github.com/googlearchive/android-Camera2Video
    
### (2) Object Detection YOLOv4

    https://github.com/hunglc007/tensorflow-yolov4-tflite
    
### (3) Beacon 設置
    
    https://github.com/dburr/linux-ibeacon

