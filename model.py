import torch
import torchvision.models as models

# 1. 下載預訓練的 ResNet18
model = models.resnet18(pretrained=True)
model.eval()

# 2. 建立假資料 (Batch size=1, Channels=3, Height=224, Width=224)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 轉出 ONNX
torch.onnx.export(model, dummy_input, "resnet18.onnx",
                  input_names=['input_tensor'],
                  output_names=['output_tensor'])

print("成功產生 resnet18.onnx！")