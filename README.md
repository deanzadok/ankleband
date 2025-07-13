# Smart Ankleband for Plug-and-Play Hand-Prosthetic Control

Official code repository for [https://arxiv.org/abs/2503.17846](https://arxiv.org/abs/2503.17846).  
See our demo in action by watching the video [https://youtu.be/IUvm3WCvYG8](https://youtu.be/IUvm3WCvYG8).  

<img src="materials/demo.gif" alt="robotic_hand_demo" width="900"/>  

*A demo showing our smart ankleband with the robotic hand to perform a daily activity, such as pouring a liquid from a soda can into a paper cup.*

### Dataset  

Our dataset is publicly available! Please refer to this page for explanations and instructions on how to download:  
[Dataset explanation](DATASET.md)

### System Requirements  

* Operating system - The project was built using Ubuntu 22.04, but should work on Windows as well.
* GPU - Any Nvidia RTX GPU is sufficient (the model is tiny and require approximately 500MB of GPU memory).
* RAM - 32GB or higher is required for training purposes.

The robotic hand we used for this project is based on an opensource project provided by Haifa3D, a non-profit organization providing self-made 3d-printed solutions for those who need them. To rebuild the hand, the STL files and code to program the hand are in the following repositories:  
[https://github.com/Haifa3D/hand-mechanical-design](https://github.com/Haifa3D/hand-mechanical-design)  
[https://github.com/Haifa3D/hand-electronic-design](https://github.com/Haifa3D/hand-electronic-design)  

### Installation  
  
The project is based on Python 3.9 and PyTorch 2.6 with CUDA 12.4. All the necessary packages are in ```requirements.txt```. We recommend creating a virtual environment using Anaconda as follows:  
  
1) Download and install Anaconda Python from here:  
[https://www.anaconda.com/products/individual/](https://www.anaconda.com/products/individual/)  
  
2) Enter the following commands to create a virtual environment:  
```
conda create -n imugr python=3.9
conda activate imugr
pip install -r requirements.txt
```

For more information on how to manage conda environments, please refer to:  
[https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)  

### Overview  
  
Training and testing-related files are found in the trainer folder, where they are operated using configuration (JSON) files that are organized in the config directory. The data folder contains a data loader that is executed for both training and testing (```load_data.py```), and the evaluation folder contains all the necessary code to replicate the experiments presented in the paper (given outputs from the testing procedures).  

The model presented in the paper is found in ```trainer/models/conv1d_model.py``` under the ```Conv1DNet``` class, and the baselines are implemented in the ```train_classic.py``` that is in the same folder.  

**Training**  

All training sessions are executed using the same training file (```trainer/train_conv.py```), by calling different JSON files. An example of training the model on our dataset is as follows (in some situations we referred to the ankleband as a bracelet):

```  
python trainer/train_conv.py --json config/bracelet/regular_bracelet_leaveone.json  
```  

In case needed, you can overwrite some of the properties listed in the JSON file. For example, if we want to exclude subject 7 and leave it for testing (instead of subject no. 1), we can execute the script as follows:

```  
python trainer/train_conv.py --json config/bracelet/regular_bracelet_leaveone.json  --loo 7
``` 

However, these JSON files contain all the necessary properties and there is no need to feed the python script with additional arguments.  

**Testing**  

While some of the evaluation is done during training and the training scripts are programmed to write test metrics into the output folder, additional script files are provided to further evaluate the methods. For example, ```trainer/compute_cm.py``` can be used to compute confusion matrix, and ```trainer/compute_metrics.py``` is provided to compute the metrics presented in the paper on an specific set. Here are a few execution examples:

```  
python trainer/compute_cm.py --json config/bracelet/test_regular_bracelet.json  
python trainer/compute_metrics.py --json config/bracelet/test_regular_bracelet.json  
```  

### Real-time Implementation  

To convert the model for execution on an ESP32 board, we wrote a script that loads our model and write all trained parameters into a single C++ file so we can recreate the inference procedure on the ESP32 board. The file ```model_conversions/extract_model_weights.py``` loads the pytorch model and writes the result as ```model_weights.h```.

To install the code on the ESP32, we uploaded the code project (along with our trained model) to the ```rt_code``` folder. The main file is ```rt_code/execute_imu_gestures.ino```, the class that manages the neural network inference is in the files ```neural_network_engine.cpp``` and ```neural_network_engine.h```. The model weights are in ```model_weights_feb27_conv_c10_im64_fc2.h``` (which is basically the result of the script file ```model_conversions/extract_model_weights.py```).

Here are the required libraries to install in the Arduino IDE in order to compile the code for ESP32:

* Adafruit BNO08x (Version 1.25).
* Eigen (Version 0.3.2).
* ArduinoBLE - should be included in the ESP32 library by Espressif. 

### Acknowledgments  

This research was supported in part by the Technion Autonomous Systems Program (TASP), the Wynn Family Foundation, the David Himelberg Foundation, the Israeli Ministry of Science \& Technology grants No. 3-17385 and in part by the United States-Israel Binational Science Foundation (BSF) grants no. 2019703 and 2021643.

We also thank the subjects that participated in the creation of the dataset, and [Haifa3D](https://www.facebook.com/Haifa3d/), a non-profit organization providing self-made 3d-printed solutions, for their consulting and support through the research.  