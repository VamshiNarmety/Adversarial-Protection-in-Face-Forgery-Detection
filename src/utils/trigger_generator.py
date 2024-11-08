import torch
import torch.nn as nn
import torch.nn.functional as F
class TriggerGenerator(nn.Module):
    def __init__(self):
        super(TriggerGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
    
    def forward(self, z):
        z = self.conv1(z)
        z = F.relu(z)
        z = self.conv2(z)
        z = F.relu(z)
        z = self.conv3(z)
        return z
    
def create_kernel(v):
    size = 2*v+1
    kernel = torch.full((size, size), -1.0)
    kernel[v, v] = ((size)*(size))-1
    kernel = kernel.expand(3, 1, size, size)
    return kernel


def trigger_loss(trigger, kernel):
    kernel = kernel.to(trigger.device)
    conv_res =  F.conv2d(trigger, kernel, stride=1, padding=1, groups=3)
    norm = torch.norm(conv_res, p=1)
    return -torch.log(norm)


def embed_trigger(x, delta, mask, a):
    device = x.device  # Get the device of input x
    delta = delta.to(device)  # Move delta to the same device
    mask = mask.to(device)
    alpha = a*x/255.0
    return x+(alpha*delta*mask)


def training_trigger_generator(generator, optimizer, num_epochs, z_dim, v):
    kernel = create_kernel(v)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        #sample from a normal distribution
        z = torch.randn(1, *z_dim)
        delta = generator(z)
        loss = trigger_loss(delta, kernel)
        loss.backward()
        optimizer.step()
        if(epoch+1)%100==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    save_path = 'venv/src/utils/generator.pth'
    torch.save(generator.state_dict(), save_path)
    print(f'Model saved to {save_path}')
        
if __name__=='__main__':
    z_dim = (3, 224, 224)
    v=1
    num_epochs=1000
    generator = TriggerGenerator()
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    training_trigger_generator(generator, optimizer, num_epochs, z_dim, v)
    