import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from data_loader import get_dataset

#selecting device as per GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv(nn.Module):
    
    def __init__(self, kernel_size, stride=1, padding=True, kernel_tensor=None):
        super(Conv, self).__init__()
        self.init_params(kernel_size, stride, padding, kernel_tensor)
    
    #Function to initialize nn model params
    def init_params(self, kernel_size, stride, padding, kernel_tensor):
            
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            
            self.padding_layer = nn.ZeroPad2d(self.kernel_size//2)
            
            if kernel_tensor is None:
                self.kernel = nn.Parameter(torch.randint(0, 2, (self.kernel_size, self.kernel_size), dtype=torch.float))
            else:
                self.kernel = nn.Parameter(kernel_tensor)
                
            
            
    #forward pass of the nn model
    
    def forward(self, image):
        
        if self.padding:
            padded_image = self.padding_layer(image)
        else:
            padded_image = image
        
        stop = self.kernel_size - 1
        target_shape = padded_image.shape[-2] - stop, padded_image.shape[-1] - stop
        
        conv_imgs = torch.zeros((image.shape[0], image.shape[1], target_shape[0], target_shape[1])).to(device)
        
        for n in range(padded_image.shape[0]):
            for c in range(padded_image.shape[1]):
                conv_img = list()
                temp_img = padded_image[n][c]
                for i in range(padded_image.shape[2] - stop):
                    for j in range(padded_image.shape[3] - stop):
                        temp_img_view = temp_img[i : i + self.kernel_size, j : j + self.kernel_size]
                        conv_img.append(torch.sum(torch.mul(temp_img_view, self.kernel)))
                conv_img = torch.stack(conv_img)
                conv_img = torch.reshape(conv_img, target_shape)
                conv_img = torch.unsqueeze(conv_img, 0)
            conv_imgs[n] = conv_img
        return conv_imgs


#Test Case for Conv Forward pass

# if __name__ == "__main__":
#     image = torch.tensor([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]], dtype=torch.float)
#     image = torch.unsqueeze(image, 0)
#     image = torch.unsqueeze(image, 0)
#     kernel = torch.tensor([[1, 0, 1],[0, 1, 0],[1, 0, 1]], dtype=torch.float)
#     conv = Conv(3, kernel_tensor=kernel)
#     print(conv(image))

#     # check with PyTorch implementation
#     kernel = torch.unsqueeze(kernel, 0)
#     kernel = torch.unsqueeze(kernel, 0)
#     print(get_torch_conv(image, kernel, 1))
