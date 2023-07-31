# BTS_cpp

#### Thanks to
[BTS](https://github.com/cleinc/bts)

## Environment
 * Ubuntu 20.04.3 LTS
 * ROS noetic 1.15.14
 * GCC 10.3.0
 * CMake 3.22.2
 * LibTorch 1.9.1
 * CUDA 11.1
 * OpenCV 4.2.0
 
## Download Large Files
[Download](https://o365cbnu-my.sharepoint.com/:u:/g/personal/2019132001_cbnu_ac_kr/ES0GPFV8I8pHnr8LmZd_I3ABNgdrchMxoSgWl248G39EtA?e=eqlknF)  
[Tree](./inference_ros/src/inference_bts/data)

## Package Folder Tree
BTS_cpp
 * [train](./train)
   * [build](./train/build)
   * [src](./train/src)
     * [glob](./train/src/glob)
     * [bts](./train/src/bts)
 * [inference_ros](./inference_ros)
     * [src](./inference_ros/src)
         * [inference_bts](./inference_ros/src/inference_bts)
             * [src](./inference_ros/src/inference_bts/src)
               * [bts](./inference_ros/src/inference_bts/src/bts)
             * [launch](./inference_ros/src/inference_bts/launch)
             * [data](./inference_ros/src/inference_bts/data)
