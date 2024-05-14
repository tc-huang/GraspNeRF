import cv2
import numpy as np
import time
import pyrealsense2 as rs

font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)

def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        S = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / S
        x = (R[2, 1] - R[1, 2]) * S
        y = (R[0, 2] - R[2, 0]) * S
        z = (R[1, 0] - R[0, 1]) * S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return np.array([w, x, y, z])

def Img_ArUco_detect(img,Intrinsic,distortion):
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(cv2.aruco_dict , parameters)
    corners, ids, rejected_img_points = detector.detectMarkers(img)
    #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    #corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

#    如果找不打id
    if ids is not None:
        #邊長單位兩公分,但我以m為基準所以是0.02 m
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, Intrinsic, distortion)
        # 估计每个标记的姿态并返回值rvet和tvec ---不同
        # from camera coeficcients
        (rvec-tvec).any() # get rid of that nasty numpy value array error

#        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1) #绘制轴
#        aruco.drawDetectedMarkers(frame, corners) #在标记周围画一个正方形

        for i in range(rvec.shape[0]):
            cv2.drawFrameAxes(img, Intrinsic, distortion, rvec, tvec, 0.05) 
            cv2.aruco.drawDetectedMarkers(img, corners)
        ###### DRAW ID #####
        cv2.putText(img, "Id: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        r_mat,_ = cv2.Rodrigues(rvec)  

        print("-----------------")
        print(r_mat)
        print("tvec:")
        print(tvec)
        quaternion = rotation_matrix_to_quaternion(r_mat)
        print("四元數")
        print(quaternion)
        print("-----------------")
    else:
        ##### DRAW "NO IDS" #####
        cv2.putText(img, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

    key = cv2.waitKey(1)

    if key == 27:         # 按esc键退出
        print('esc break...')
        cv2.destroyAllWindows()

    if key == ord(' '):   # 按空格键保存
#        num = num + 1
#        filename = "frames_%s.jpg" % num  # 保存一张图像
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, img)
    cv2.waitKey(0)

def Camstream_ArUco_detect():
    # 生成aruco标记
    # 加载预定义的字典
    dist=np.array([ 4.90913179e-02 , 5.22699002e-01, -2.65209452e-03  ,1.13033224e-03,-2.17133474e+00])

    mtx=np.array([[606.77737126 ,  0.      ,   321.63287183],[  0.     ,    606.70030146 ,236.95293136],[  0.    ,       0.    ,       1.        ]])

    # mtx=np.array([[398.12724231  , 0.      ,   304.35638757],
    #  [  0.       ,  345.38259888, 282.49861858],
    #  [  0.,           0.,           1.        ]])


    # 初始化RealSense相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 开始采集
    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # 将帧数据转换为OpenCV格式
        frame = np.asanyarray(color_frame.get_data())
        cv2.imshow('RealSense Color Image', frame)

        # 检测按键，如果按下q键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        h1, w1 = frame.shape[:2]
        # 读取摄像头画面
        # 纠正畸变
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
        # dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # x, y, w1, h1 = roi
        # dst1 = dst1[y:y + h1, x:x + w1]
        # frame=dst1


        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        parameters =  cv2.aruco.DetectorParameters()
        # dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        '''
        detectMarkers(...)
            detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
            mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''

        detector = cv2.aruco.ArucoDetector(cv2.aruco_dict , parameters)
        corners, ids, rejected_img_points = detector.detectMarkers(frame)
        #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
        #corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

    #    如果找不打id
        if ids is not None:
            #邊長單位兩公分,但我以m為基準所以是0.02 m
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, mtx, dist)
            # 估计每个标记的姿态并返回值rvet和tvec ---不同
            # from camera coeficcients
            (rvec-tvec).any() # get rid of that nasty numpy value array error

    #        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1) #绘制轴
    #        aruco.drawDetectedMarkers(frame, corners) #在标记周围画一个正方形

            for i in range(rvec.shape[0]):
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.05) 
                cv2.aruco.drawDetectedMarkers(frame, corners)
            ###### DRAW ID #####
            cv2.putText(frame, "Id: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            r_mat,_ = cv2.Rodrigues(rvec)  

            print("-----------------")
            print(r_mat)
            print("tvec:")
            print(tvec)
            quaternion = rotation_matrix_to_quaternion(r_mat)
            print("四元數")
            print(quaternion)
            print("-----------------")
        else:
            ##### DRAW "NO IDS" #####
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


        # 显示结果框架
        cv2.imshow("frame",frame)

        key = cv2.waitKey(1)

        if key == 27:         # 按esc键退出
            print('esc break...')
            cv2.destroyAllWindows()
            pipeline.stop()
            break

        if key == ord(' '):   # 按空格键保存
    #        num = num + 1
    #        filename = "frames_%s.jpg" % num  # 保存一张图像
            filename = str(time.time())[:10] + ".jpg"
            cv2.imwrite(filename, frame)
        cv2.waitKey(0)

if __name__ == "__main__":
    Camstream_ArUco_detect()