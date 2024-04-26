from model import *
from prepare_dataset import *
import torchvision

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform).data
trainloader = prepare_dataset(trainset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DiffusionModel(T=4000, embed_size=64, n_channels=1, n_classes=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.555))
criterion = nn.HuberLoss()
epochs = 50
for epoch in range(epochs):
    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        dict_, noise = batch
        X_noisy = dict_["X_noisy"].to(device)
        time = dict_["time"].to(device)

        output = model(X_noisy, time).squeeze(1).unsqueeze(-1)
        noise.to(device)

        optimizer.zero_grad()
        loss = criterion(output, noise)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        if i % 200 == 199:
            print(f'Epoch {epoch+1}/{i}, loss: {running_loss}')
            running_loss = 0.0
            
