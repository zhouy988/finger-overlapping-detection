# import cv2
# import numpy as np
# import mediapipe as mp
# import random
# import time
# import os
# import json

# # 初始化MediaPipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# # 配置参数
# DATA_DIR = "overlap_dataset"
# TOTAL_SAMPLES = 10  # 需要收集的样本数
# FINGERS = ["thumb1", "index2", "middle3", "ring4", "pinky5"]  # 所有手指

# # 手指关键点索引（MediaPipe 21个关键点）
# FINGER_TIPS = {
#     "thumb": 4,
#     "index": 8,
#     "middle": 12,
#     "ring": 16,
#     "pinky": 20
# }

# def generate_random_instruction():
#     """随机生成手指交叠指令"""
#     f1, f2 = random.sample(FINGERS, 2)
#     return f"{f1}>{f2}", f1, f2

# def record_sample(cap, instruction, top_finger, bottom_finger):
#     """记录一个样本的数据"""
#     start_time = time.time()
#     sample_data = {
#         "instruction": instruction,
#         "top_finger": top_finger,
#         "bottom_finger": bottom_finger,
#         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "landmarks": None
#     }
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue
        
#         # 显示指令和倒计时
#         cv2.putText(frame, instruction, (50, 50), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.putText(frame, f"采集倒计时: {3 - (time.time() - start_time):.1f}s", 
#                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
#         # 处理手部检测
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(image_rgb)
        
#         if results.multi_hand_landmarks:
#             hand_landmarks = results.multi_hand_landmarks[0]
            
#             # 绘制手部关键点
#             mp.solutions.drawing_utils.draw_landmarks(
#                 frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
#             # 3秒后自动记录
#             if time.time() - start_time > 3:
#                 # 保存所有关键点的3D坐标
#                 sample_data["landmarks"] = [
#                     {"x": lm.x, "y": lm.y, "z": lm.z} 
#                     for lm in hand_landmarks.landmark
#                 ]
#                 cv2.putText(frame, "已记录!", (200, 200), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
#                 cv2.imshow("Data Collection", frame)
#                 cv2.waitKey(5000)  # 显示0.5秒提示
#                 return sample_data
        
#         cv2.imshow("Data Collection", frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
#             return None

# def main():
#     # 创建数据目录
#     os.makedirs(DATA_DIR, exist_ok=True)
#     dataset = []
    
#     cap = cv2.VideoCapture(0)
    
#     print(f"将收集{TOTAL_SAMPLES}个手指交叠样本...")
    
#     for i in range(TOTAL_SAMPLES):
#         # 生成随机指令
#         instruction, top_finger, bottom_finger = generate_random_instruction()
#         print(f"\n样本 {i+1}/{TOTAL_SAMPLES}: {instruction}")
        
#         # 记录样本
#         sample = record_sample(cap, instruction, top_finger, bottom_finger)
#         if sample:
#             dataset.append(sample)
#             print(f"已记录: {top_finger}在上, {bottom_finger}在下")
#         else:
#             print("用户中断")
#             break
    
#     # 保存数据
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(DATA_DIR, f"overlap_data_{timestamp}.json")
#     with open(filename, 'w') as f:
#         json.dump(dataset, f, indent=2)
#     print(f"\n数据已保存到 {filename}")
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import mediapipe as mp
import random
import time
import os
import json

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Configuration parameters
DATA_DIR = "overlap_dataset"
TOTAL_SAMPLES = 500       # Total number of samples to collect
SAMPLES_PER_INTERVAL = 50 # Number of samples per interval
BREAK_DURATION = 15       # Seconds to break before next interval
FINGERS = ["thumb1", "index2", "middle3", "ring4", "pinky5"]  # All fingers

# Mapping for finger tip landmarks (MediaPipe uses 21 landmarks)
FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20
}

def generate_random_instruction():
    """Generate a random instruction for finger overlap."""
    f1, f2 = random.sample(FINGERS, 2)
    return f"{f1}>{f2}", f1, f2

def record_sample(cap, instruction, top_finger, bottom_finger):
    """Record one sample of data."""
    start_time = time.time()
    sample_data = {
        "instruction": instruction,
        "top_finger": top_finger,
        "bottom_finger": bottom_finger,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "landmarks": None
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Display instruction and countdown
        cv2.putText(frame, instruction, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Collection countdown: {3 - (time.time() - start_time):.1f}s", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Process hand detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Automatically record after 3 seconds
            if time.time() - start_time > 3:
                # Save all landmark 3D coordinates
                sample_data["landmarks"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z} 
                    for lm in hand_landmarks.landmark
                ]
                cv2.putText(frame, "Recorded!", (200, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(500)  # Show the recorded message for 0.5 seconds
                return sample_data
        
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            return None

def main():
    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset = []
    
    cap = cv2.VideoCapture(0)
    
    print(f"Collecting {TOTAL_SAMPLES} finger overlap samples...")
    
    for i in range(TOTAL_SAMPLES):
        # Generate a random instruction
        instruction, top_finger, bottom_finger = generate_random_instruction()
        print(f"\nSample {i+1}/{TOTAL_SAMPLES}: {instruction}")
        
        # Record the sample
        sample = record_sample(cap, instruction, top_finger, bottom_finger)
        if sample:
            dataset.append(sample)
            print(f"Recorded: {top_finger} is on top, {bottom_finger} is at bottom")
        else:
            print("User interrupted")
            break
        
        # Pause for a break after every interval of samples (except after the last interval)
        if (i+1) % SAMPLES_PER_INTERVAL == 0 and (i+1) < TOTAL_SAMPLES:
            print(f"\nCollected {i+1} samples so far. Taking a {BREAK_DURATION}-second break before the next interval...")
            time.sleep(BREAK_DURATION)
    
    # Save the dataset to a JSON file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f"overlap_data_{timestamp}.json")
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"\nData saved to {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
