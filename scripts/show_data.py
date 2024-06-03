import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytransform3d.transform_manager import TransformManager
import sys
sys.path.append('src')

# from pysurroundcap.util import *

if __name__ == "__main__":
    sim_images = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/new_sim_images.npy')
    sim_extrinsics = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/new_sim_extrinsics.npy')
    sim_intrinsics = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/new_sim_intrinsics.npy')
    sim_depth_range = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/new_sim_depth_range.npy')
    # volume = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/volume.npy')
    sim_bbox3d = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/new_sim_bbox3d.npy')
    """
    images: (6, 3, 288, 512)
    extrinsicsn:
    [[-8.6602539e-01  5.0000000e-01 -4.8147158e-16  6.7502973e-16]
    [ 4.0957603e-01  7.0940650e-01 -5.7357645e-01  2.1646656e-16]
    [-2.8678823e-01 -4.9673176e-01 -8.1915206e-01  5.0000000e-01]]
    intrinsics:
    [[357.048   0.    255.8  ]
    [  0.    357.048 143.8  ]
    [  0.      0.      1.   ]]
    depth_range
    [[0.2 0.8]
    [0.2 0.8]
    [0.2 0.8]
    [0.2 0.8]
    [0.2 0.8]
    [0.2 0.8]]
    bbox3d
    [[-0.15   -0.15   -0.0503]
    [ 0.15    0.15    0.2497]]
    """
    # real_extrinsics = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/real_extrinsics.npy')
    # my_intrinsics = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/my_intrinsics.npy')
    # my_images = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/my_images.npy')

    # ee_poses = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/ee_poses.npy')
    # print(ee_poses)

    # camera_pose = np.load('/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/data/traindata_example/giga_hemisphere_train_demo/packed_full/packed_0-170/000fc0562d2a4881b24921a424ef9175/camera_pose.npy')
    # print()
    # print(f"camera_pose:\n{camera_pose.shape}")
    # print(f"camera_pose inv:\n{np.linalg.inv(camera_pose[0])}")
    # print(f"real_extrinsicsn:\n{real_extrinsics[0]}")
    print(f"images: {sim_images.shape} {sim_images.min()} {sim_images.max()}")
    print(f"extrinsicsn:\n{sim_extrinsics[0]}")
    print(f"intrinsics:\n{sim_intrinsics[0]}")
    print(f"depth_range\n{sim_depth_range}")
    # print(f"volume\n{volume.shape}")
    print(f"bbox3d\n{sim_bbox3d}")

    # print(f"my_intrinsics:\n{my_intrinsics[0]}")

    
    

    for i in range(6):
        img = sim_images[i].transpose((1, 2, 0))
        img = (img * 255).astype(np.uint8)
        cv2.imshow(f'sim_image[{i}]', img)
        # cv2.imwrite(f'sim_image[{i}].png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)
    for i in range(6):
        img = my_images[i].transpose((1, 2, 0))
        cv2.imshow(f'my_image_[{i}]', img)
    
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)
    
    tm = TransformManager()

    for i in range(6):
        # M = np.linalg.inv(np.vstack([sim_extrinsics[i], np.array([0, 0, 0, 1])]))
        M = np.vstack([sim_extrinsics[i], np.array([0, 0, 0, 1])])
        # M = np.linalg.inv(M)
        # print(f'M: {T_to_xyzrpy(M)[-3:]}')
        # print(f'M.inv: {np.linalg.inv(M)}')
        # print(f'M[:3, 3]: {M[:3, 3]}')
        # tm.add_transform(f"pt", f"r[{i}]", np.linalg.inv(M))
        tm.add_transform(f"ws", f"s[{i}]", M)

    print()
    # exit(0)

    for i in range(6):
        # T = ee_poses[i]
        T = np.linalg.inv(ee_poses[i])
        tm.add_transform(f"base", f"e[{i}]", T)

    print()
    # exit(0)


    for i in range(6):
        # N = np.linalg.inv(real_extrinsics[i])
        N = real_extrinsics[i]
        # N = np.linalg.inv(N)
        # N[0, 3] += 100 / 1000
        # N[1, 3] += 100 / 1000
        # N[2, 3] += 100
        # N[:3, 3] /= 1000
        # # print(f'N[:3, 3]: {N[:3, 3]}')
        # N[:3, 3] = np.array([0.5, 0.5, 0.5]).T
        # print(f'N: {T_to_xyzrpy(N)[-3:]}')
        tm.add_transform(f"ws", f"r[{i}]", N)
        break
        # K = np.array(
        #     [
        #         [1, 0, 0, 0.5],
        #         [0, 1, 0, 0.5],
        #         [0, 0, 1, 0.5],
        #         [0, 0, 0, 1],
        #     ]
        # )
        # tm.add_transform(f"pt", f"t", np.linalg.inv(K))

    # ax = tm.plot_frames_in("ws", s=0.1)
    # ax.set_xlim((-0.5, 0.5))
    # ax.set_ylim((-0.5, 0.5))
    # ax.set_zlim((0.0, 0.6))
    # ax = tm.plot_frames_in("ws", s=0.2)
    ax = tm.plot_frames_in("base", s=100)
    ax.set_xlim((-500, 500))
    ax.set_ylim((-500, 500))
    ax.set_zlim((0.0, 500))
    plt.show()