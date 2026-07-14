import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AnomalyDataset
from models import DINOv2Teacher, CustomStudentCNN

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Models
    teacher = DINOv2Teacher(size='large').to(device)
    student = CustomStudentCNN(out_channels=1024).to(device)

    # Dataset and Dataloader
    # Training only on normal data
    train_dataset = AnomalyDataset(args.data_path, is_train=True, image_size=518)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels, _ in pbar:
            images = images.to(device)
            
            with torch.no_grad():
                teacher_features = teacher(images)
                # Normalize teacher features over channel dimension
                teacher_features = nn.functional.normalize(teacher_features, dim=1, p=2)
                
            student_features = student(images)
            # Normalize student features
            student_features = nn.functional.normalize(student_features, dim=1, p=2)
            
            loss = criterion(student_features, teacher_features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), os.path.join(args.save_dir, 'best_student.pth'))
            print("Saved new best model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset", help="Path to the dataset root")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save weights")
    args = parser.parse_args()
    
    train(args)
