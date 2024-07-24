import torch
from module import SnakeNet
import random



def predict(model, data_channel, device: torch.device = 'cpu') -> str:
    # 将输入数据转换为 PyTorch 张量，并移动到指定设备
    input_data = torch.tensor(data_channel, dtype=torch.float32).unsqueeze(0).to(device)
    print(input_data.shape)

    # 确保模型在指定设备上，并设置为评估模式
    model.to(device)
    model.eval()

    # 禁用梯度计算（我们不需要梯度在推理时）
    with torch.no_grad():
        output_data = model(input_data)

    # 处理输出数据，根据输出的最大值选择方向
    directions = ['up', 'down', 'left', 'right']
    predicted_index = torch.argmax(output_data, dim=1).item()
    predicted_direction = directions[predicted_index]
    return predicted_direction

def train(model, optimizer, loss_fn, memory, gamma, batch_size, device):
    if len(memory) < batch_size:
        return  # 如果样本数量不足，返回等待

    # 从经验回放缓冲区中随机采样一个批次
    batch = random.sample(memory, batch_size)
    
    # 分离批次中的数据
    state_batch = torch.cat([b[0] for b in batch]).to(device)
    action_batch = torch.tensor([b[1] for b in batch], device=device)
    reward_batch = torch.tensor([b[2] for b in batch], device=device, dtype=torch.float32)
    next_state_batch = torch.cat([b[3] for b in batch]).to(device)
    done_batch = torch.tensor([b[4] for b in batch], device=device, dtype=torch.float32)
    
    # 获取当前 Q 值
    current_q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
    
    # 计算目标 Q 值
    with torch.no_grad():
        next_q_values = model(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)
    
    # 计算损失
    loss = loss_fn(current_q_values, target_q_values)
    
    # 优化模型参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


