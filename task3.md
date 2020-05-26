字符识别模型
# 1.CNN基础与原理
卷积神经网络（Convolutional Neural Network）CNN实质上是协方差计算，并不是数学意义上的卷积运算。

每层通过卷积核对输入像素进行卷积运算。随着层数增加，图像持续逐渐减小，卷积核的感受野变大（即对应于原图上，能够影响输出值的区域变大）。

CNN包括卷积层、池化层（Maxpooling,AveragePooling）、ReLU（非线性激活层）、DropOut、全连接层等构成。

# 2.CNN的发展

从LeNet、AlexNet、VGG16/19、GoogLeNet、NIN、Inception v1-v4、ResNet到后来的轻量级网络ShuffleNet、MobileNet、EfficientNet,再到NAS，CNN网络已逐渐成熟。

# 3.利用Pytorch构建CNN模型

网络采用了ResNet18预训练模型，并增加了5个并列的全连接层用于预测五位数字的每位数字。

# 4.代码实践
	import torchvision.models as models
	import torchvision.transforms as transforms
	import torchvision.datasets as datasets
	import torch.nn as nn
	from torch.utils import model_zoo
	import torch


	class SVHN_Model1(nn.Module):
    	def __init__(self):
        	super(SVHN_Model1, self).__init__()

        	model_conv = models.resnet18(pretrained=False)
        	model_conv.load_state_dict(torch.load('../weights/resnet18-5c106cde.pth)
        	model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        	model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        	self.cnn = model_conv

      	    self.fc0=nn.Linear(512,6)

    	    self.fc1 = nn.Linear(512, 11)
     	    self.fc2 = nn.Linear(512, 11)
 	        self.fc3 = nn.Linear(512, 11)
    	    self.fc4 = nn.Linear(512, 11)
     	    self.fc5 = nn.Linear(512, 11)

    	def forward(self, img):
        	feat = self.cnn(img)
        	# print(feat.shape)
        	feat = feat.view(feat.shape[0], -1)
        	c0 = self.fc0(feat)
       	    c1 = self.fc1(feat)
        	c2 = self.fc2(feat)
        	c3 = self.fc3(feat)
        	c4 = self.fc4(feat)
        	c5 = self.fc5(feat)
        	return c0, c1, c2, c3, c4, c5
