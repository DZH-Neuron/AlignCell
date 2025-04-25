import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict(model, data, device=device):
    """模型预测"""
    model.eval()
    data = torch.tensor(data).to(device)
    with torch.no_grad():
        outputs, _ = model(data)
    return outputs().to(device)
