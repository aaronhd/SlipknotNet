#!/usr/bin/env python
import copy
import math
from torchvision import transforms
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import cv2

# import importlib.util
import tkinter as tk
import numpy as np
import time
from collections import Counter
from models import get_network


# # Establish connection with Davinci
# pyc_file = "/home/yaoting/catkin_ws/src/davinci/src/clutchsolution/__pycache__/stopMovement.cpython-310.pyc"
# # Load the module from the .pyc file
# spec = importlib.util.spec_from_file_location("StopMovement", pyc_file)
# module = importlib.util.module_from_spec(spec)
# sys.modules["stop_movement"] = module
# spec.loader.exec_module(module)


# # 初始化全局变量
# da_Vinci = None
# connection_status = False

# # 初始化 控制类并建立连接
# # 实例化
# da_Vinci = module.StopMovement(com_port="/dev/ttyACM0", baud_rate=9600)
# da_Vinci.buildConnection()
# try:
#     connection_status = True
#     # 建立连接
#     da_Vinci.buildConnection()
# except Exception as e:
#     print(f"连接失败: {e}")
#     connection_status = False

# # 用来存储倒计时的计时器 ID
# countdown_id = None


def resize_image(image, width=1920, height=1080, resized_used=True):
    if resized_used:
        new_size = (width, height)
        resized_image = cv2.resize(image, new_size)
        return resized_image
    else:
        return image


def post_process(output):
    output[output < 0] = 0.0

    cmap = cm.get_cmap("gray")
    output = cmap(output)[:, :, 0]

    return output


def get_std_var(keypoints):
    # 1. 计算中心点（均值点）
    center = np.mean(keypoints, axis=0)  # 对 x 和 y 分别求均值
    # print(f"Center Point: {center}")

    # 2. 计算每个特征点到中心点的距离
    distances = np.linalg.norm(keypoints - center, axis=1)  # 欧几里得距离
    distances = np.round(distances, 2)
    print(f"Distances to Center: {distances}")

    # median_distance = np.median(distances)
    # mad = np.median(np.abs(distances - median_distance))

    # # 设置阈值（例如：中位数 ± 3 倍 MAD）
    # mad_threshold = 3 * mad
    # filtered_keypoints = keypoints[np.abs(distances - median_distance) <= mad_threshold]

    # 3. 计算距离的方差和标准差
    variance = np.var(distances)  # 方差
    std_dev = np.std(distances)  # 标准差

    # z_score = (keypoints-center)/std_dev

    # print('z_score', z_score)
    return std_dev, distances


def video_stream():
    global knot_opening
    global da_Vinci
    global pull_stage
    global send_done
    global endoscope_used

    if endoscope_used:
        # # Get endoscope video stream 打开视频流，假设是/dev/video0设备
        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        desired_fps = 60
        cap.set(cv2.CAP_PROP_FPS, desired_fps)
        # 获取实际帧率
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Actual FPS: {actual_fps}")

        if not cap.isOpened():
            print(
                "Failed to capture frame, give the permission sudo chmod 777 /dev/video2"
            )
            return

        # 设置视频格式为YUY2
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "2"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置宽度为 1080p
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置高度为 1080p
        cap.set(cv2.CAP_PROP_FPS, 30)

    flash_num = 0
    while True:
        if endoscope_used:
            if knot_opening is False:
                ret, frame = cap.read()
                flash_num += 1

            if not ret:
                print(
                    "Failed to capture frame, give the permission sudo chmod 777 /dev/video2"
                )
                break

            bgr_frame = frame
            bgr_frame = bgr_frame[136:854, 259:1463, :]
            bgr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            (
                raw_h,
                raw_w,
                _,
            ) = bgr_frame.shape
            image_raw = Image.fromarray(bgr_frame)
        else:
            # offline test data
            flash_num += 1
            image_path = "demo/exp_demo1.png"
            # image_path = 'demo/exp_demo2.png'

            image_raw = Image.open(image_path).convert("RGB")  # 加载 RGB 图像
            bgr_frame = np.array(image_raw)
            raw_h, raw_w, _ = bgr_frame.shape
            # image_raw = image_raw.crop((830,383, 1917, 1074)) # xyxy top left right lower

        template_image_path = "demo/black_t_11.png"  # yiyuan  ok  scale 0.3
        template_image = cv2.imread(
            template_image_path, cv2.IMREAD_GRAYSCALE
        )  # 模板图像

        scale = 0.8
        template_image = cv2.resize(template_image, None, fx=scale, fy=scale)

        # 获取 template_image 和 vis_bgr_frame 的尺寸
        h1, w1 = template_image.shape[:2]
        h2, w2 = bgr_frame.shape[:2]
        # print(h1, w1 , h2, w2 ) # 96 243 718 1204

        # 计算画布的高度（取两者中较大的高度）
        max_height = max(h1, h2)
        template_image_3ch = cv2.cvtColor(template_image, cv2.COLOR_GRAY2BGR)

        # # 创建黑色背景的画布，宽度为 w1 + w2，高度为 max_height
        # result_image_0 = np.ones((max_height, w1 + w2, 3), dtype=np.uint8)
        # 创建浅灰色背景的画布，宽度为 w1 + w2，高度为 max_height，通道数为 3
        light_gray = (192, 192, 192)  # 浅灰色的 RGB 值
        result_image_0 = np.full((max_height, w1 + w2, 3), light_gray, dtype=np.uint8)

        # 将 template_image 放置在画布左侧
        result_image_0[:h1, :w1] = template_image_3ch

        # 将 vis_bgr_frame 放置在画布右侧
        vis_bgr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        result_image_0[:h2, w1 : w1 + w2] = vis_bgr_frame
        if flash_num % 20 > 5:
            cv2.circle(
                result_image_0,
                (
                    w1 + 50,
                    int(h1 / 2) + 10,
                ),
                20,
                (0, 255, 0),
                -1,
            )

        cv2.imshow("Suture Detection", result_image_0)

        transform = transforms.Compose(
            [
                transforms.Resize((800, 800)),  # 调整图像大小
                transforms.ToTensor(),  # 转换为 Tensor 格式
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet均值和标准差
            ]
        )

        # transform0 = transforms.Compose(
        #     [
        #         transforms.Resize((800, 800)),  # 调整图像大小
        #         transforms.ToTensor(),  # 转换为 Tensor 格式

        #     ]
        # )
        start_time = time.time()

        # tensor_image_raw = transform0(image_raw)
        # tensor_image_raw = tensor_image_raw.unsqueeze(0)

        image = transform(image_raw)
        image = image.unsqueeze(0)

        xc = image.to(device)
        repeat = 1
        outputs = np.zeros((800, 800))
        # outputs_list = []
        for _ in range(repeat):
            with torch.no_grad():
                output_tmp = net(xc)
            output_tmp = output_tmp.cpu().squeeze().detach().numpy()  # (800, 800)
            output_tmp = post_process(output_tmp)
            outputs += output_tmp
            # outputs_list.append(output_tmp)
        end_time = time.time()
        # print('time cost:', end_time-start_time)
        output = outputs / repeat
        # output = np.maximum(outputs_list[0], outputs_list[1])
        # print(output.shape)
        # knot_opening = False

        mask = np.uint8(resize_image(output, raw_w, raw_h) * 255)
        # _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        # # print(mask.shape)
        # cv2.imshow('mask frame', mask)

        edges = cv2.Canny(mask, 50, 150)

        # 霍夫变换检测线段
        # lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=160, maxLineGap=10)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=160, maxLineGap=50
        )

        # 创建一个彩色图像用于绘制结果
        output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 如果检测到线段
        length_list = []
        x_min_list = []
        x_max_list = []
        y_min_list = []
        y_max_list = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                length_list.append(length)
                # print(length)
                # 计算扩展后的边界框坐标
                padding = 5  # 扩展像素，根据需要调整
                x_min = max(min(x1, x2) - padding, 0)
                y_min = max(min(y1, y2) - padding, 0)
                x_max = min(max(x1, x2) + padding, mask.shape[1])
                y_max = min(max(y1, y2) + padding, mask.shape[0])

                tan_vaule = (y_max - y_min) / float(x_max - x_min)
                # print(tan_vaule)
                if tan_vaule > 0.201:
                    continue

                x_min_list.append(x_min)
                x_max_list.append(x_max)
                y_min_list.append(y_min)
                y_max_list.append(y_max)

                # 绘制矩形边界框
                cv2.rectangle(
                    output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
                )
                # 可选：绘制检测到的线段
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 显示结果
        # if len(length_list) > 0 and max(length_list) > 380:
        if len(length_list) > 0 and max(length_list) > 290:
            pull_stage = True
        else:
            pull_stage = False

        def get_crop_infor(x_min_list, x_max_list, y_min_list, y_max_list):
            x1 = min(x_min_list)
            y1 = min(y_min_list)

            x2 = max(x_max_list)
            y2 = max(y_max_list)

            # print(x1,y1, x2,y2)

            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)

            origin_w = abs(x1 - x2)
            origin_h = abs(y1 - y2)

            # w = int(origin_w)
            w = int(max(300, origin_w))
            h = int(max(160, origin_h))  # 260 150 100

            new_x1 = max(int(center_x - 0.5 * w), 0)
            new_y1 = max(int(center_y - 0.5 * h), 0)
            new_x2 = min(int(center_x + 0.5 * w), 1023)
            new_y2 = min(int(center_y + 0.5 * h), 717)

            return (new_x1, new_y1, abs(new_x1 - new_x2), abs(new_y1 - new_y2)), (
                new_x1,
                new_y1,
                new_x2,
                new_y2,
            )

        if len(length_list) > 0:
            cv2.putText(
                img=output_image,
                text="max line: " + str(int(max(length_list))),
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1,
                color=(255, 100, 0),
                thickness=1,
            )
        # cv2.imshow('Mask with Bounding Boxes', output_image)
        pull_stage = 1
        if pull_stage:
            try:
                xywh, xyxy = get_crop_infor(
                    x_min_list, x_max_list, y_min_list, y_max_list
                )
            except:
                continue
            # print(xywh)
            crop_bgr_frame = copy.deepcopy(bgr_frame)
            # vis_bgr_frame = copy.deepcopy(bgr_frame)
            crop_bgr_frame = crop_bgr_frame[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2], :]
            # crop_mask = mask[xyxy[1]: xyxy[3], xyxy[0]:xyxy[2]]
            crop_mask = mask

            cv2.rectangle(
                vis_bgr_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2
            )

            try:
                cv2.imshow("realtime mask", crop_mask)
                # cv2.imshow('bgr_frame', bgr_frame)
                crop_bgr_frame = cv2.cvtColor(crop_bgr_frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("crop_bgr_frame", crop_bgr_frame)
            except:
                print("No mask or crop_bgr frame")
                continue

            sift_time = time.time()

            # 2. 初始化 SIFT 检测器
            sift = cv2.SIFT_create(
                nfeatures=1000,  # 保留 500 个最佳特征点
                nOctaveLayers=3,  # 每组金字塔中的层数
                contrastThreshold=0.01,  # 对比度阈值   0.04
                edgeThreshold=20,  # 边缘阈值  10  20
                sigma=1.4,  # 高斯模糊初始值 1.6
            )

            # 3. 检测关键点和计算描述符
            keypoints1, descriptors1 = sift.detectAndCompute(template_image, None)
            # keypoints2, descriptors2 = sift.detectAndCompute(mask, None)
            keypoints2, descriptors2 = sift.detectAndCompute(crop_mask, None)

            if descriptors2 is None or descriptors1 is None:
                continue
            if len(descriptors2) == 0 or len(descriptors1) == 0:
                continue

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            # 5. 过滤匹配点（使用 Lowe's ratio test）
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:  # Lowe's ratio test  0.7
                        good_matches.append(m)
            except:
                print(len(matches))
            end_time = time.time()
            print(f"找到 {len(good_matches)} 个匹配点")

            print("time cost:", end_time - start_time)
            print("sift time cost:", end_time - sift_time)

            # Extract the coordinates of the good matches
            points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
            points2 = [keypoints2[m.trainIdx].pt for m in good_matches]

            std_var = -1
            if len(points2) > 2:
                std_var, distance_list = get_std_var(points2)
                counter_dict = Counter(distance_list)
                print(counter_dict)
                print(counter_dict.values())
                print(max(counter_dict.values()))
                print("std_var: ", std_var)

            if len(good_matches) > 4:
                if (
                    std_var < 11
                    and max(distance_list) < 50
                    and min(distance_list) > 0
                    and max(counter_dict.values()) < 3
                ):
                    knot_opening = True

            # 9. 将所有 keypoints2 转换到裁剪前的坐标系下
            keypoints2_original = []
            for kp in keypoints2:
                # 将关键点坐标加上裁剪区域的偏移量 (x, y)
                pt = (kp.pt[0] + xywh[0], kp.pt[1] + xywh[1])
                # 创建一个新的关键点对象，保留原始关键点的其他属性（如大小和方向）
                keypoints2_original.append(
                    cv2.KeyPoint(
                        pt[0],
                        pt[1],
                        kp.size,
                        kp.angle,
                        kp.response,
                        kp.octave,
                        kp.class_id,
                    )
                )

                result_image = cv2.drawMatches(
                    template_image,
                    keypoints1,
                    vis_bgr_frame,
                    keypoints2,
                    good_matches,
                    None,
                    flags=2,
                )
            # result_image = cv2.drawMatches(template_image, keypoints1, mask, keypoints2, good_matches, None, flags=2)
            grey_background = np.full((h2 - h1, w1, 3), light_gray, dtype=np.uint8)
            result_image[h1:h2, 0:w1, :] = grey_background
            print(f"找到 {len(good_matches)} 个匹配点")
            # 7. 显示匹配结果
            cv2.putText(
                img=result_image,
                text="strightline detect",
                org=(10, h1 + 40),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.8,
                color=(255, 100, 0),
                thickness=1,
            )

            if knot_opening:
                cv2.circle(
                    result_image,
                    (
                        w1 + 50,
                        int(h1 / 2) + 10,
                    ),
                    20,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    img=result_image,
                    text="knot is opening...",
                    org=(10, h1 + 80),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=0.8,
                    color=(255, 100, 0),
                    thickness=1,
                )
            else:
                if flash_num % 20 > 5:
                    cv2.circle(
                        result_image,
                        (
                            w1 + 50,
                            int(h1 / 2) + 10,
                        ),
                        20,
                        (0, 255, 0),
                        -1,
                    )

            cv2.putText(
                img=result_image,
                text="feature num: "
                + str(int(len(good_matches)))
                + "  "
                + str(std_var),
                org=(10, h2 - 30),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.8,
                color=(255, 100, 0),
                thickness=1,
            )

            cv2.imshow("Suture Detection", result_image)

        if knot_opening:
            print("knot is opening !")
            if send_done == False:
                # da_Vinci.stopMovement()
                send_done = True

        # cv2.imshow('real time frame', bgr_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if endoscope_used:
        # 释放视频流
        cap.release()


if __name__ == "__main__":
    network_name = "slipknotnet"
    model_path = "checkpoint/2025-02-26_01-46-57_sutureresnet_/epoch_50"

    input_channels = 3
    slipknotnet = get_network(network_name)
    print("input_channels", input_channels)
    net = slipknotnet(input_channels=input_channels)

    net.load_state_dict(torch.load(model_path, map_location="cuda:0"))

    device = torch.device("cuda:0")
    net = net.to(device)
    net.eval()

    knot_opening = False
    send_done = False
    pull_stage = False
    endoscope_used = False
    peak_list = []

    video_stream()
