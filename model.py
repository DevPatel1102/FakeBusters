import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import json
from torchvision import models
from facenet_pytorch import MTCNN
from concurrent.futures import ThreadPoolExecutor
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EfficientNetWithSE(nn.Module):
    def _init_(self):
        super(EfficientNetWithSE, self)._init_()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.efficientnet(x)

mtcnn = MTCNN(keep_all=True, device=device)

def preprocess_frames(frames, batch_size=16):
    processed_batches = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(preprocess_batch, frames[i:i+batch_size])
                   for i in range(0, len(frames), batch_size)]
        for future in futures:
            processed_batches.extend(future.result())
    return processed_batches

def preprocess_batch(batch):
    processed = []
    for frame in batch:
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                processed.append(preprocess_image(face))
    return processed

def preprocess_image(frame):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(frame)

def extract_frames(video_path, sample_interval=10):
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    frame_count = 0

    while success:
        if frame_count % sample_interval == 0:
            frames.append(frame)
        success, frame = video.read()
        frame_count += 1

    video.release()
    return frames

def predict_deepfake(frames, model, step=5):
    model.eval()
    processed_frames = preprocess_frames(frames)
    total_fake_prob = 0
    count = 0

    for i in range(0, len(processed_frames), step):
        batch = torch.stack(processed_frames[i:i + step]).to(device)
        with torch.no_grad():
            outputs = model(batch).cpu()
        total_fake_prob += outputs.sum().item()
        count += len(outputs)

    average_fake_prob = total_fake_prob / count
    return average_fake_prob-0.2

def classify_video(fake_prob):
    if fake_prob > 0.5:
        print(f"The video is FAKE with {fake_prob * 100:.2f}% confidence.")
        return 1
    else:
        print(f"The video is REAL with {(1 - fake_prob) * 100:.2f}% confidence.")
        return 0

efficientnet_model = EfficientNetWithSE().to(device)

def deepfake_detection():
    video_path = input("Enter the path to the video file: ")

    start_time = time.time()
    frames = extract_frames(video_path)
    print(f"Frames extracted in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    fake_prob = predict_deepfake(frames, model=efficientnet_model)
    print(f"Deepfake detection completed in {time.time() - start_time:.2f} seconds.")
    
    model_output = classify_video(fake_prob)

    model_output_json = {"video_id": video_path, "model_prediction": model_output}
    with open('model_output.json', 'w') as json_file:
        json.dump(model_output_json, json_file)
    print("Model output saved to model_output.json")

    user_input = int(input("Please authenticate the video (0 for real, 1 for fake): "))
    user_output_json = {"video_id": video_path, "user_authentication": user_input}
    with open('user_input.json', 'w') as json_file:
        json.dump(user_output_json, json_file)
    print("User input saved to user_input.json")

    if model_output != user_input:
        print("Mismatch detected. Retraining the model...")
        retrain_model(model=efficientnet_model, frames=frames, label=user_input)
    else:
        print("No mismatch detected.")

def retrain_model(model, frames, label):
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Batch inference and retraining
    processed_frames = preprocess_frames(frames)
    labels = torch.tensor([label] * len(processed_frames)).float().to(device).unsqueeze(1)

    for epoch in range(1):
        total_loss = 0
        for i in range(0, len(processed_frames), 8):
            batch = torch.stack(processed_frames[i:i+8]).to(device)
            label_batch = labels[i:i+8]

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Retraining epoch completed with loss: {total_loss:.4f}")

    torch.save(model.state_dict(), 'fine_tuned_model.pth')
    print("Model retrained and saved as fine_tuned_model.pth")

deepfake_detection()