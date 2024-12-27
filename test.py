import torch
import torch.nn as nn
from torchmetrics import Accuracy

# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=16)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=16, out_features=16)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

# 创建模型实例并移动到设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleModel().to(device)

# 示例输入数据
x_train = torch.randn(10, 2).to(device)
y_train = torch.randint(0, 2, (10, 1)).float().to(device)

# 计算输出
output = model(x_train)
print("Logits:", output)

# 预测概率
probs = torch.sigmoid(output)
print("Pred probs:", probs)

# 预测标签
pred_labels = (probs > 0.5).float()
print("Pred labels:", pred_labels)

# 计算准确率
accuracy = Accuracy(num_classes=2, multiclass=True).to(device)
acc = accuracy(pred_labels, y_train)
print("Accuracy:", acc.item())