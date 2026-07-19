import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SkeletonDataset
from model import STGCN
from sklearn.metrics import classification_report, confusion_matrix # pip install scikit-learn

def test_model():
    NUM_CLASSES = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = SkeletonDataset(data_dir='data/test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = STGCN(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('best_stgcn_weights.pth', map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n=== ST-GCN Test Classification Metrics Report ===")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    
    print("=== Target Confusion Matrix ===")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    test_model()
