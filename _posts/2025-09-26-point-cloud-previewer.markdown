---
layout: post
title:  "Point Cloud Previewer"
date:   2025-09-26 11:00:00 -0400
categories: 3D
---
I implemented a simple point cloud previewer with fly mode navigation and camera switching using open3D. The code is available on [Github](https://github.com/ronchuxia/PointCloudPreviewer).

# Background
I have been training 3D Gaussian Splatting (3dgs) on my own dataset recently. However, the dataset I used is not in the standard format required by 3dgs. More specifically, the camera extrinsics are not aligned with the point cloud. I need to identify the issue and align the camera with the scene.

I have been using some online 3dgs or point cloud previewers. However, most of them:
- do not support visualizing both point clouds and camera frustums
- do not support fly mode navigation
- do not support camera switching for quick preview

Therefore, I decided to implement a simple point cloud previewer with the above features.

# Coordinate Systems

I choose to use Open3D for point cloud importing and visualization. However, COLMAP and Open3D use different coordinate systems. This requires conversion when importing camera parameters from COLMAP to Open3D.

OpenCV / COLMAP:
- Right-handed coordinate system
- X: Right (points to right of image)
- Y: Down (points to bottom of image)
- Z: Forward (into the scene, along viewing direction)

Open3D:
- Right-handed coordinate system
- X: Right (points to right of image)
- Y: Up (points to top of image)
- Z: Backward (away from the scene, against viewing direction)

When talking about coordinate systems of a library, we usually refer to the **camera coordinate system**.

For the **world coordinate system**, both COLMAP and Open3D use a right-handed coordinate system. The up / down, left / right, forward / backward directions are defined by the user or the scene.

By convention, we use RGB to represent the three axes of a right-handed coordinate system:
- Red: X axis
- Green: Y axis
- Blue: Z axis

In summary, the difference between (the camera coordinate frames of) COLMAP and Open3D is that the Y and Z axes are flipped. 

# Camera Pose and Extrinsics

Both Camera Pose and Camera Extrinsics are $4 \times 4$ matrices that describe camera and world coordinate transformations.
- Camera Pose: C2W (Camera to World)
- Camera Extrinsics: W2C (World to Camera)

C2W is the inverse of W2C.

Based on the definition:
- `C2W[:3, 3]` is the camera center in world coordinates.
- `C2W[:3, 0]` is the camera's x axis in world coordinates. For COLMAP cameras, it points to the right of the image.
- `C2W[:3, 1]` is the camera's y axis in world coordinates. For COLMAP cameras, it points to the bottom of the image.
- `C2W[:3, 2]` is the camera's z axis in world coordinates. For COLMAP cameras, it points into the scene, along the viewing direction.

Based on this, we can implement the fly mode camera navigation by extracting and updating the camera pose.

# C2W Conversion

Because of the difference in coordinate systems, to convert a COLMAP C2W matrix to an Open3D C2W matrix, we need to flip the Y and Z axes of the camera frame. This can be done by multiplying the C2W matrix with a flip matrix:

$$\text{flip} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$

The Open3D C2W matrix is equivalent to the doing the following operations:
1. Flip the y axis and z axis of the camera frame to convert from Open3D camera frame to COLMAP camera frame.
2. Apply the original COLMAP C2W matrix to convert from COLMAP camera frame to COLMAP / Open3D world frame.

# 3DGS PLY Format

3DGS uses a custom PLY format to store point clouds, which is slightly different from the standard PLY format. 

The normal in 3DGS PLY is set to zero, which prevents Open3D from rendering the colored point cloud. Also, the color in 3DGS PLY is sometimes stored as SH coefficients instead of RGB values. Therefore, I implement a simple PLY parser to read the point cloud position and convert the color to RGB if required. The normals are discarded.

# Implementation

I implemented this program with the help of Claude Code. I have never used Open3D before, and Claude Code saved me the trouble of learning the basics and reading the documentation. However, I still need to debug and figure out some details by myself. For example, the zero normal issue mentioned above, and the coordinate system conversion.

In my experience, currently generative AI often fail at tasks like coordinate system conversion etc. Maybe it's because of different conventions and notations used in existing literature and libraries, and sometimes requires some spatial reasoning, which is not very easy for current AI. This should be a fun direction to explore in the future.

Because of this, I reviewed the concepts of coordinate systems and camera extrinsics / pose, and got a better understanding of these concepts through writing this program.

# TODO

This is a simple point cloud previewer. There are many things to improve:
- Add support for 3DGS rendering
    - There are already some implementations using WebGL, which is a cross-platform web technology. I can add fly mode navigation and camera switching to these implementations.
- More intuitive camera control
    - The current camera control is very basic. I can add mouse or trackpad control.

Luckily, the event-driven architecture of open3D is very convenient for customizing camera control. It can also be easily transferred to other event-driven frameworks for extended functionality.

# Coming Up Next
I have been using Claude Code to help me with various tasks recently, not just coding. It has been very helpful and efficient. In the next blog I plan to share some experience with prompting and utilizing Claude Code.


