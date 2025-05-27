import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class UrbanSoundDataset(Dataset):
    def __init__(self, dataframe, audio_dir, transform=None, target_sample_rate=16000, n_mels=64, max_len=128):
        self.dataframe = dataframe
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.max_len = max_len

        self.resampler = T.Resample(orig_freq=44100, new_freq=self.target_sample_rate)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        file_path = os.path.join(self.audio_dir, f"fold{row['fold']}", row['slice_file_name'])
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != self.target_sample_rate:
            waveform = self.resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

        mel_spec = self.pad_or_trim(mel_spec)

        return mel_spec, row['classID']

    def pad_or_trim(self, spec):
        time_dim = spec.shape[-1]
        if time_dim > self.max_len:
            spec = spec[..., :self.max_len]
        elif time_dim < self.max_len:
            pad_amt = self.max_len - time_dim
            spec = torch.nn.functional.pad(spec, (0, pad_amt))
        return spec


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)


def load_data(metadata_path, audio_dir, batch_size=16):
    df = pd.read_csv(metadata_path)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['classID'])

    train_dataset = UrbanSoundDataset(train_df, audio_dir)
    test_dataset = UrbanSoundDataset(test_df, audio_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def main():
    metadata_path = 'UrbanSound8K/metadata/UrbanSound8K.csv'
    audio_dir = 'UrbanSound8K/audio'
    batch_size = 16
    epochs = 10
    learning_rate = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader = load_data(metadata_path, audio_dir, batch_size)

    model = AudioClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, device, epochs)
    evaluate(model, test_loader, device)

    #Download the weights using the code below
    #torch.save(model.state_dict(), 'audio_classifier.pth')


if __name__ == '__main__':
    main()
