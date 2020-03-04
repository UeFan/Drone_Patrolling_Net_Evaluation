## Learn by Observation: Imitation Learning for Drone Patrolling from Videos of A Human Navigator
This is the code that we use to train and evaluate.
The following figures are the the evaluation results. (first row: DroNet [1]; second row: TrailNet [2]; Third row: our UAVPatrolNet )

![image,img](images/20200104_171745.jpg)
## Get Start

### Requirements
This code has been tested on Ubuntu 18.04, and on Python 3.6.  


Dependencies:
* TensorFlow 1.15.0
* Keras 2.2.4
* Keras-contrib 2.0.8
* tensorflow-estimator         
* tensorflow-probability          
* tensorflow-tensorboard
* NumPy
* OpenCV
* scikit-learn
* Python gflags

### Usage
1. Run the test code, there are `predict_UAVPatrolNet.py`,`predict_DroNet.py`and`predict_TrailNet.py`
    
    ```
    python predict_whichNet.py [Flags]
    ```
    
    Check configuration parameters in `common_flags.py`  
2. Run the accuracy calculation code, there are`evaluate_UAVPatrolNet.py`,`evaluate_DroNet.py`and`evaluate_TrailNet.py`
    
    ```
    python evaluate_whichNet.py [Flags]
    ```
3. Format of data set.
    ```
    validation_dir/
        direction/
            images/
            direction_n_filted.txt
        translation/
            images/    
            translation.txt
        ...
    test_dir/
        testingset1/
            images/
    ```
    
    Reference:
    
    [1]. A. Loquercio, A. I. Maqueda, C. R. Del-Blanco, and D. Scaramuzza, “Dronet: Learning to fly by driving,” IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 1088–1095, 2018.
    
    [2]. N. Smolyanskiy, A. Kamenev, J. Smith, and S. Birchfield, “Toward low-flying autonomous mav trail navigation using deep neural networks for environmental awareness,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2017, pp. 4241–4247.
