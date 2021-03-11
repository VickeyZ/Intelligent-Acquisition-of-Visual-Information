# Intelligent-Acquisition-of-Visual-Information
Intelligent Acquisition of Visual Information Course

## Course Contents
1. Introduction
2. Modern Camera Systems
3. Lightfields & Image Relighting
4. Measuring Depths
5. Volumetric Acquisition
6. Appearance Capture
7. Advanced Topics

## Course Targets
1. Image Processing
2. Camera Calibrations
3. Shape Modeling & Fabrication
4. Stereo Computation
5. Structured Lighting
6. BRDF Acquisition
7. Final Project

### Lab1
- Study the relationship between camera parameters and captured images(pixels) via experiments:
  1.Exposure
  2.Gain
- [Bonus 25%] Derive a quantitative noise model for captured pixels
  1.What are related parameters?
  2.What is the equation of the noise?
  3.How well it explains captured pixels?
- Download and compile the starter code that will allow you to directly operate on captured images

### Lab2
- Single Camera Calibration with OpenCV
- You may use your own camera if its focal length can be fixed
- Recommend using a chessboard on a Pad / monitor
- Goals
  1. Finish intrinsic and extrinsic calibration
  2. Discuss the impact of different factors (e.g., # of images, coverage of the chessboard, view angle, etc) over the final reprojection error
  3. Output the estimated camera centers and the chessboard into a .ply file for 3D viewing in software like MeshLab
  4. Project some interesting 3D points on the input images (i.e., augmented reality)
  5. [Bonus 15%] Make a real-time demo. You may want to fix the intrinsic parameters and solve for the extrinsic ones only.

### Lab3
1. Perform stereo calibration with OpenCV sample
2. Compute a dense depth map / 3D point cloud from two calibrated input views via triangulation
    - Evaluate the impact of different parameters (e.g., patch size) over final quality
3. Resolve the color discrepancy between two views and produce the final colored 3D point cloud, along with the cameras, in a single .ply file
    - You may want to perform radiometric calibration first.

### Lab4 Projector-Camera-Based Stereo Vision
Approach #1: 2 Cameras + 1 Projector  
  Stereo calibrate the two cameras  
  Project a pattern / a few patterns  
  Establish the correspondences between stereo calibrated cameras  
  Triangulate (Depth map & 3D point cloud)  
Approach #2: 1 Camera + 1 Projector  
  Calibrate the camera-projector system  
  http://mesh.brown.edu/calibration/  
  Project a pattern / a few patterns  
  Establish the correspondences between camera pixels and projector pixels  
  Triangulate (Depth map & 3D point cloud)  
As approach 2 has been chosen.  

### Course Project
Intro: 通过单个相机完成对相机前人物姿势的识别（称为主控相机），从而控制另一个相机（称为被控相机）进行各种操作，包括但不限于启动快门，调整常用的相机参数等操作。
