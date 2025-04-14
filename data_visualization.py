import plotly.graph_objects as go
import mediapipe as mp
import json

# 加载数据
with open('overlap_dataset/overlap_data_20250414_092037.json') as f:
    data = json.load(f)

def plot_overlap_3d(sample):
    # 提取关键点坐标
    landmarks = sample['landmarks']
    x = [lm['x'] for lm in landmarks]
    y = [lm['y'] for lm in landmarks]
    z = [lm['z'] for lm in landmarks]
    
    # 获取上下手指的指尖索引
    finger_indices = {'thumb':4, 'index':8, 'middle':12, 'ring':16, 'pinky':20}
    top_idx = finger_indices[sample['top_finger'].rstrip('1234')]  # 移除末尾数字
    bottom_idx = finger_indices[sample['bottom_finger'].rstrip('1234')]
    
    fig = go.Figure()
    
    # 绘制所有关键点（灰色）
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color='lightgray'),
        name='All joints'
    ))
    
    # 高亮上手指（红色）
    fig.add_trace(go.Scatter3d(
        x=[x[top_idx]], y=[y[top_idx]], z=[z[top_idx]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name=f"Top: {sample['top_finger']}"
    ))
    
    # 高亮下手指（蓝色）
    fig.add_trace(go.Scatter3d(
        x=[x[bottom_idx]], y=[y[bottom_idx]], z=[z[bottom_idx]],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name=f"Bottom: {sample['bottom_finger']}"
    ))
    
    # 添加连线
    connections = mp.solutions.hands.HAND_CONNECTIONS
    for conn in connections:
        fig.add_trace(go.Scatter3d(
            x=[x[conn[0]], x[conn[1]]],
            y=[y[conn[0]], y[conn[1]]],
            z=[z[conn[0]], z[conn[1]]],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))
    
    # 添加上下手指连线（虚线）
    fig.add_trace(go.Scatter3d(
        x=[x[top_idx], x[bottom_idx]],
        y=[y[top_idx], y[bottom_idx]],
        z=[z[top_idx], z[bottom_idx]],
        mode='lines',
        line=dict(color='purple', width=2, dash='dash'),
        name='Overlap'
    ))
    
    fig.update_layout(
        title=f"Overlap: {sample['instruction']}",
        scene=dict(
            xaxis=dict(title='X', range=[0.4,0.9]),
            yaxis=dict(title='Y', range=[0.4,0.9]),
            zaxis=dict(title='Z', range=[-0.05,0.05]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.show()

# 可视化第一个样本
plot_overlap_3d(data[0])