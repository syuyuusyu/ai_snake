import torch
from module import SnakeNet
import random



def predict(model, data_channel, device: torch.device = 'cpu',epsilon = 0.1) -> str:
    directions = ['up', 'down', 'left', 'right']
    if random.random() < epsilon:
        # 探索：随机选择一个动作
        return random.choice(directions)
    # 将输入数据转换为 PyTorch 张量，并移动到指定设备
    input_data = format_data(data_channel,device)

    # 确保模型在指定设备上，并设置为评估模式
    model.to(device)
    model.eval()

    # 禁用梯度计算（我们不需要梯度在推理时）
    with torch.no_grad():
        output_data = model(input_data)

    # 处理输出数据，根据输出的最大值选择方向

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

def format_data(play_ground,device:torch.device='cpu'):
    input_data = torch.tensor(play_ground, dtype=torch.float32)
    # 如果需要调整 x 和 y 的顺序，可以使用 .transpose() 或 .permute()
    input_data = input_data.transpose(0, 1)  # 如果 play_ground 是 (x, y)，这一步转换为 (y, x)
    # 增加 batch 维度和 channel 维度
    input_data = input_data.unsqueeze(0).unsqueeze(1)
    return input_data.to(device)


def save_checkpoint(model, optimizer, filepath):
    state = {

        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer, device):
    # 加载检查点并将其移动到指定设备
    checkpoint = torch.load(filepath, map_location=device)
    
    # 加载模型状态字典并将模型移动到指定设备
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 加载优化器状态字典并确保它们在同一设备上
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 如果需要，可以将优化器中的状态参数也移动到正确的设备上
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
