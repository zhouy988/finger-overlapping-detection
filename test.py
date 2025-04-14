import cv2
import mediapipe as mp

# 初始化MediaPipe Hands模块
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 打开摄像头
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("忽略空摄像头帧")
            continue
        
        # 转换颜色空间 BGR to RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理手势检测
        results = hands.process(image)
        
        # 绘制检测结果
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点和连接线
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # 水平翻转图像以获得自拍视图
        image = cv2.flip(image, 1)
        
        # 显示提示信息
        cv2.putText(image, "按 'P' 打印坐标 | ESC退出", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('MediaPipe Hands', image)
        
        key = cv2.waitKey(5)
        # 按ESC退出
        if key & 0xFF == 27:
            break
        # 按P打印坐标
        elif key & 0xFF == ord('p') or key & 0xFF == ord('P'):
            if results.multi_hand_landmarks:
                print("\n=== 手部关键点坐标 ===")
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    print(f"\n手 #{hand_idx + 1}:")
                    for landmark_idx, landmark in enumerate(hand_landmarks.landmark):
                        print(f"点 {landmark_idx}: (X: {landmark.x:.4f}, Y: {landmark.y:.4f}, Z: {landmark.z:.4f})")
                print("=====================\n")
            else:
                print("当前帧未检测到手部！")

# 释放资源
cap.release()
cv2.destroyAllWindows()