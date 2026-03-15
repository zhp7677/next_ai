# 第一部分：导入库和基础设置
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from collections import defaultdict
import pandas as pd

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据路径
DATA_PATH = r'F:\next_ai\data'
SAVE_PATH = r'F:\next_ai\models'
os.makedirs(SAVE_PATH, exist_ok=True)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 第二部分：CIFAR-100数据加载
def get_cifar100_loaders(batch_size=128, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    try:
        trainset = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=True, download=False, transform=transform_train
        )
    except:
        print(f"Downloading CIFAR-100 to {DATA_PATH}...")
        trainset = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=True, download=True, transform=transform_train
        )
    
    try:
        testset = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=False, download=False, transform=transform_test
        )
    except:
        testset = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=False, download=True, transform=transform_test
        )
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, testloader, trainset, testset

trainloader, testloader, trainset, testset = get_cifar100_loaders(batch_size=128)
print(f"Train samples: {len(trainset)}, Test samples: {len(testset)}")

# 第三部分：TrackedCNN - 精确的下游激活统计（保留空间对应）
class TrackedCNN(nn.Module):
    """
    统计每个卷积核（神经元）在正确分类样本中是否成功激活下一层的神经元。
    成功激活的定义：下一层的某个输出通道的最终输出有正值（被激活），
    并且该输入通道对该输出通道的独立卷积贡献也在至少一个空间位置上有正值。
    """
    def __init__(self, num_classes=100):
        super(TrackedCNN, self).__init__()
        
        # 卷积层配置
        self.conv_configs = [
            ('conv1', 3, 64, 3, 1, 1),
            ('conv2', 64, 128, 3, 1, 1),
            ('conv3', 128, 256, 3, 1, 1),
            ('conv4', 256, 256, 3, 1, 1),
            ('conv5', 256, 512, 3, 1, 1),
            ('conv6', 512, 512, 3, 1, 1),
            ('conv7', 512, 512, 3, 1, 1),
            ('conv8', 512, 512, 3, 1, 1),
        ]
        
        # 创建卷积层和BN层
        for name, in_ch, out_ch, k, s, p in self.conv_configs:
            setattr(self, name, nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
            setattr(self, f'bn{name[-1]}', nn.BatchNorm2d(out_ch))
        
        # 全连接层
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # 存储每个样本中，上一层神经元对下一层的贡献掩码以及下一层的激活状态
        self.downstream_data = {}  # 键: 上一层名, 值: (contrib_mask, output_activated)
                                  # contrib_mask: (batch, C_in, C_out) 布尔，表示输入通道是否对输出通道有正贡献
                                  # output_activated: (batch, C_out) 布尔，表示输出通道是否最终激活
        
        # 神经元统计（仅卷积层）
        self.neuron_stats = {
            'downstream_correct_count': defaultdict(int),   # 被下一层至少一个神经元成功激活的次数
            'downstream_total_activations': defaultdict(int), # 成功激活的下一层神经元总数（累加）
            # ====== 新增 ======
            'downstream_full_activation_count': defaultdict(int) # 成功激活下一层所有神经元的次数
        }
        
        # 记录卷积层名称列表
        self.conv_names = [cfg[0] for cfg in self.conv_configs]
        
    def get_filter_id(self, layer_name, channel):
        return f"{layer_name}_{channel}"
    
    def _compute_downstream_activation(self, layer_name, x_in, conv_layer, output):
        """
        对于当前层 conv_layer，其输入为 x_in（来自上一层），输出为 output（经BN+ReLU后）。
        计算每个输入通道（即上一层神经元）对每个输出通道的贡献掩码，以及输出通道本身的激活掩码。
        返回：
            contrib_mask: (batch, C_in, C_out) 布尔张量
            output_activated: (batch, C_out) 布尔张量
        """
        weight = conv_layer.weight  # (C_out, C_in, kH, kW)
        stride = conv_layer.stride
        padding = conv_layer.padding
        dilation = conv_layer.dilation
        C_out, C_in, kH, kW = weight.shape
        batch_size = x_in.size(0)
        device = x_in.device

        # 初始化贡献掩码为 False
        contrib_mask = torch.zeros(batch_size, C_in, C_out, dtype=torch.bool, device=device)

        for c_in in range(C_in):
            w_slice = weight[:, c_in:c_in+1, :, :]  # (C_out, 1, kH, kW)
            x_slice = x_in[:, c_in:c_in+1, :, :]    # (batch, 1, H, W)
            out_slice = F.conv2d(x_slice, w_slice, stride=stride, padding=padding, dilation=dilation)  # (batch, C_out, H', W')
            # 判断该输入通道在每个输出通道的每个空间位置是否有正值
            has_act = (out_slice > 0)  # (batch, C_out, H', W')
            # 对于每个输出通道，只要有一个空间位置有正值，就认为该输入通道对该输出通道有贡献
            contrib_mask[:, c_in, :] = has_act.any(dim=(2, 3))

        # 计算输出通道本身的激活状态：最终输出特征图至少有一个空间位置 >0
        output_activated = (output > 0).any(dim=(2, 3))  # (batch, C_out)

        return contrib_mask.cpu(), output_activated.cpu()
    
    def forward(self, x, track_activations=False):
        """
        前向传播，如果 track_activations 为 True，则记录每一层对上一层的精确激活信息。
        """
        self.downstream_data = {}
        current = x

        # Conv1
        if track_activations:
            x_conv1_in = current
        current = F.relu(self.bn1(self.conv1(current)))
        # conv1 没有上一层，不统计

        # Conv2 + Pool1
        if track_activations:
            x_conv2_in = current  # 即 conv1 的输出
        current = F.relu(self.bn2(self.conv2(current)))
        if track_activations:
            contrib, out_act = self._compute_downstream_activation('conv1', x_conv2_in, self.conv2, current)
            self.downstream_data['conv1'] = (contrib, out_act)
        current, _ = F.max_pool2d(current, 2, 2, return_indices=True)

        # Conv3
        if track_activations:
            x_conv3_in = current  # conv2 的输出（池化后）
        current = F.relu(self.bn3(self.conv3(current)))
        if track_activations:
            contrib, out_act = self._compute_downstream_activation('conv2', x_conv3_in, self.conv3, current)
            self.downstream_data['conv2'] = (contrib, out_act)

        # Conv4 + Pool2
        if track_activations:
            x_conv4_in = current
        current = F.relu(self.bn4(self.conv4(current)))
        if track_activations:
            contrib, out_act = self._compute_downstream_activation('conv3', x_conv4_in, self.conv4, current)
            self.downstream_data['conv3'] = (contrib, out_act)
        current, _ = F.max_pool2d(current, 2, 2, return_indices=True)

        # Conv5
        if track_activations:
            x_conv5_in = current
        current = F.relu(self.bn5(self.conv5(current)))
        if track_activations:
            contrib, out_act = self._compute_downstream_activation('conv4', x_conv5_in, self.conv5, current)
            self.downstream_data['conv4'] = (contrib, out_act)

        # Conv6 + Pool3
        if track_activations:
            x_conv6_in = current
        current = F.relu(self.bn6(self.conv6(current)))
        if track_activations:
            contrib, out_act = self._compute_downstream_activation('conv5', x_conv6_in, self.conv6, current)
            self.downstream_data['conv5'] = (contrib, out_act)
        current, _ = F.max_pool2d(current, 2, 2, return_indices=True)

        # Conv7
        if track_activations:
            x_conv7_in = current
        current = F.relu(self.bn7(self.conv7(current)))
        if track_activations:
            contrib, out_act = self._compute_downstream_activation('conv6', x_conv7_in, self.conv7, current)
            self.downstream_data['conv6'] = (contrib, out_act)

        # Conv8 + Pool4
        if track_activations:
            x_conv8_in = current
        current = F.relu(self.bn8(self.conv8(current)))
        if track_activations:
            # 统计 conv7 对 conv8 的贡献
            contrib, out_act = self._compute_downstream_activation('conv7', x_conv8_in, self.conv8, current)
            self.downstream_data['conv7'] = (contrib, out_act)

        # 池化
        current, _ = F.max_pool2d(current, 2, 2, return_indices=True)  # 此时 current 形状 (batch, 512, 2, 2)
        pooled = current  # 保留池化后的特征图用于 fc1 统计

        # Flatten and FC
        flat = pooled.view(pooled.size(0), -1)
        fc1_out = self.fc1(flat)
        fc1_activated = F.relu(fc1_out)  # (batch, 512)

        # ====== 新增：统计 conv8 对 fc1 的贡献 ======
        if track_activations:
            # 将 fc1 的权重视为卷积核 (C_out, C_in, 2, 2)
            weight_fc1 = self.fc1.weight.view(512, 512, 2, 2)  # (512, 512, 2, 2)
            C_out, C_in, kH, kW = weight_fc1.shape
            batch_size = pooled.size(0)
            device = pooled.device

            # 初始化贡献掩码
            contrib_mask = torch.zeros(batch_size, C_in, C_out, dtype=torch.bool, device=device)

            for c_in in range(C_in):
                w_slice = weight_fc1[:, c_in:c_in+1, :, :]  # (512, 1, 2, 2)
                x_slice = pooled[:, c_in:c_in+1, :, :]      # (batch, 1, 2, 2)
                out_slice = F.conv2d(x_slice, w_slice)      # (batch, 512, 1, 1)
                has_act = (out_slice > 0).squeeze(-1).squeeze(-1)  # (batch, 512)
                contrib_mask[:, c_in, :] = has_act

            # 输出激活状态：fc1 的每个神经元是否被激活（即 ReLU 后 > 0）
            output_activated = (fc1_activated > 0)  # (batch, 512)

            self.downstream_data['conv8'] = (contrib_mask.cpu(), output_activated.cpu())

        # 继续前向
        current = fc1_activated
        current = self.fc2(current)
        return current

    def update_stats_for_correct(self, batch_labels, batch_predictions):
        """
        对于 batch 中预测正确的样本，更新每个神经元的统计。
        精确规则：对于上一层的每个神经元（输入通道 c_in），
        统计下一层中那些同时满足：
          - 输出通道被激活（output_activated 为 True）
          - 该输入通道对该输出通道有正贡献（contrib_mask 中对应位置为 True）
        的输出通道数量。累加这些数量到 downstream_total_activations，
        并且如果至少有一个这样的输出通道，则 downstream_correct_count 加1。
        ====== 新增：如果该神经元激活了下一层的所有输出通道，则 downstream_full_activation_count 加1。======
        """
        batch_size = batch_labels.size(0)
        for b in range(batch_size):
            if batch_predictions[b] != batch_labels[b]:
                continue
            # 遍历所有记录的下游数据
            for layer_name, (contrib, out_act) in self.downstream_data.items():
                # contrib: (batch, C_in, C_out) 布尔
                # out_act: (batch, C_out) 布尔
                sample_contrib = contrib[b]  # (C_in, C_out)
                sample_out_act = out_act[b]  # (C_out,)
                C_out = sample_contrib.size(1)

                # 对于每个输入通道 c_in
                C_in = sample_contrib.size(0)
                for c_in in range(C_in):
                    # 找出该输入通道有贡献且输出通道被激活的通道
                    activated_downstream = sample_contrib[c_in] & sample_out_act  # (C_out,) 布尔
                    count = activated_downstream.sum().item()
                    if count > 0:
                        neuron_id = self.get_filter_id(layer_name, c_in)
                        self.neuron_stats['downstream_correct_count'][neuron_id] += 1
                        self.neuron_stats['downstream_total_activations'][neuron_id] += count
                        # ====== 新增 ======
                        if count == C_out:
                            self.neuron_stats['downstream_full_activation_count'][neuron_id] += 1

# 创建模型
model = TrackedCNN(num_classes=100).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {total_params:,} parameters")
conv_neurons = sum(p.out_channels for p in model.modules() if isinstance(p, nn.Conv2d))
print(f"Total convolutional neurons (filters): {conv_neurons}")

# 第四部分：训练循环（与之前相同）
def train_epoch(model, trainloader, optimizer, criterion, track_every_n_batches=10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_idx = 0
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        track_activations = (batch_idx % track_every_n_batches == 0)
        
        outputs = model(inputs, track_activations=track_activations)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if track_activations:
            with torch.no_grad():
                model.update_stats_for_correct(targets, predicted)
        
        batch_idx += 1
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(trainloader)}, "
                  f"Loss: {running_loss/batch_idx:.3f}, "
                  f"Acc: {100.*correct/total:.2f}%")
    
    return running_loss/len(trainloader), 100.*correct/total

def test(model, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, track_activations=False)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss/len(testloader), 100.*correct/total

# 第五部分：保存统计结果为CSV（新增 downstream_full_activation_count 列）
def export_stats_to_csv(model, filename='neuron_downstream_stats.csv'):
    """将卷积核的下游激活统计导出为CSV"""
    data = []
    all_ids = set(model.neuron_stats['downstream_correct_count'].keys()) | \
              set(model.neuron_stats['downstream_total_activations'].keys()) | \
              set(model.neuron_stats['downstream_full_activation_count'].keys())
    for filter_id in all_ids:
        layer, channel = filter_id.rsplit('_', 1)
        correct_count = model.neuron_stats['downstream_correct_count'].get(filter_id, 0)
        total_activations = model.neuron_stats['downstream_total_activations'].get(filter_id, 0)
        full_activation_count = model.neuron_stats['downstream_full_activation_count'].get(filter_id, 0)  # 新增
        data.append({
            'layer': layer,
            'channel': int(channel),
            'neuron_id': filter_id,
            'downstream_correct_count': correct_count,
            'downstream_total_activations': total_activations,
            'downstream_full_activation_count': full_activation_count   # 新增列
        })
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(SAVE_PATH, filename), index=False)
    print(f"Exported neuron stats to {os.path.join(SAVE_PATH, filename)}")
    return df

# 第六部分：主训练循环
def main_training(num_epochs=200, lr=0.1, track_every_n_batches=10):
    print("="*80)
    print("Training with precise downstream activation counting (spatially aligned)")
    print("="*80)
    
    model = TrackedCNN(num_classes=100).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer, criterion, track_every_n_batches
        )
        
        test_loss, test_acc = test(model, testloader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%")
        
        # 打印当前统计概况
        if model.neuron_stats['downstream_correct_count']:
            nonzero = len(model.neuron_stats['downstream_correct_count'])
            total_conv_filters = sum(p.out_channels for p in model.modules() if isinstance(p, nn.Conv2d))
            print(f"Conv neurons with at least one downstream activation: {nonzero}/{total_conv_filters} ({100*nonzero/total_conv_filters:.1f}%)")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'best_model_epoch{epoch}.pth'))
        
        scheduler.step()
    
    # 训练结束，导出统计
    export_stats_to_csv(model, filename=f'neuron_downstream_stats_epoch{num_epochs}.csv')
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")
    return model, history

if __name__ == "__main__":
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.1
    TRACK_EVERY_N = 10  # 每10个batch统计一次
    
    print("Starting training with precise downstream activation counting...")
    print(f"Configuration: epochs={NUM_EPOCHS}, lr={LEARNING_RATE}, track_every={TRACK_EVERY_N}")
    
    model, history = main_training(
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        track_every_n_batches=TRACK_EVERY_N
    )