
# 1.数据读取与数据扩增

在上一章节，我们给大家讲解了赛题的内容和三种不同的解决方案。从本章开始我们将逐渐的学习使用【定长字符识别】思路来构建模型，逐步讲解赛题的解决方案和相应知识点。

本章主要内容为数据读取、数据扩增方法和Pytorch读取赛题数据三个部分组成。

-1.1 目标

   学习Python和Pytorch中图像读取

   学会扩增方法和Pytorch读取赛题数据
 
-1.2 图像读取

   由于赛题数据是图像数据，赛题的任务是识别图像中的字符。

   因此我们首先需要完成对数据的读取操作，在Python中有很多库可以完成数据读取的操作，比较常见的有Pillow和OpenCV。
   
-1.3 数据扩增方法

   在常见的数据扩增方法中，一般会从图像颜色、尺寸、形态、空间和像素等角度进行变换。当然不同的数据扩增方法可以自由进行组合，得到更加丰富的数据扩增方法。

以torchvision为例，常见的数据扩增方法包括：

transforms.CenterCrop 对图片中心进行裁剪

transforms.ColorJitter 对图像颜色的对比度、饱和度和零度进行变换

transforms.FiveCrop 对图像四个角和中心进行裁剪得到五分图像

transforms.Grayscale 对图像进行灰度变换

transforms.Pad 使用固定值进行像素填充

transforms.RandomAffine 随机仿射变换

transforms.RandomCrop 随机区域裁剪

transforms.RandomHorizontalFlip 随机水平翻转

transforms.RandomRotation 随机旋转

transforms.RandomVerticalFlip 随机垂直翻转

在本次赛题中，赛题任务是需要对图像中的字符进行识别，因此对于字符图片并不能进行翻转操作。比如字符6经过水平翻转就变成了字符9，会改变字符原本的含义。
 
-1.4 Pytorch读取数据

  由于本次赛题我们使用Pytorch框架讲解具体的解决方案，接下来将是解决赛题的第一步使用Pytorch读取赛题数据。

  在Pytorch中数据是通过Dataset进行封装，并通过DataLoder进行并行读取。所以我们只需要重载一下数据读取的逻辑就可以完成数据的读取。
  
    import os, sys, glob, shutil, json
    import cv2

    from PIL import Image
    import numpy as np

    import torch
    from torch.utils.data.dataset import Dataset
    import torchvision.transforms as transforms

    class SVHNDataset(Dataset):
        def __init__(self, img_path, img_label, transform=None):
            self.img_path = img_path
            self.img_label = img_label 
            if transform is not None:
                self.transform = transform
            else:
                self.transform = None

        def __getitem__(self, index):
            img = Image.open(self.img_path[index]).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
        
            # 原始SVHN中类别10为数字0
            lbl = np.array(self.img_label[index], dtype=np.int)
            lbl = list(lbl)  + (5 - len(lbl)) * [10]
        
            return img, torch.from_numpy(np.array(lbl[:5]))

        def __len__(self):
            return len(self.img_path)

    train_path = glob.glob('../input/train/*.png')
    train_path.sort()
    train_json = json.load(open('../input/train.json'))
    train_label = [train_json[x]['label'] for x in train_json]

    data = SVHNDataset(train_path, train_label,
              transforms.Compose([
                  # 缩放到固定尺寸
                  transforms.Resize((64, 128)),

                  # 随机颜色变换
                  transforms.ColorJitter(0.2, 0.2, 0.2),

                  # 加入随机旋转
                  transforms.RandomRotation(5),

                  # 将图片转换为pytorch 的tesntor
                  # transforms.ToTensor(),

                  # 对图像像素进行归一化
                  # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ]))
  
通过上述代码，可以将赛题的图像数据和对应标签进行读取，在读取过程中的进行数据扩增，效果如下所示：接下来我们将在定义好的Dataset基础上构建DataLoder，你可以会问有了Dataset为什么还要有DataLoder？其实这两个是两个不同的概念，是为了实现不同的功能。 Dataset：对数据集的封装，提供索引方式的对数据样本进行读取 DataLoder：对Dataset进行封装，提供批量读取的迭代读取 加入DataLoder后，数据读取代码改为如下：  
  
  
        import os, sys, glob, shutil, json
    import cv2

    from PIL import Image
    import numpy as np

    import torch
    from torch.utils.data.dataset import Dataset
    import torchvision.transforms as transforms

    class SVHNDataset(Dataset):
        def __init__(self, img_path, img_label, transform=None):
            self.img_path = img_path
            self.img_label = img_label 
            if transform is not None:
                self.transform = transform
            else:
                self.transform = None

        def __getitem__(self, index):
            img = Image.open(self.img_path[index]).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
        
            # 原始SVHN中类别10为数字0
            lbl = np.array(self.img_label[index], dtype=np.int)
            lbl = list(lbl)  + (5 - len(lbl)) * [10]
        
            return img, torch.from_numpy(np.array(lbl[:5]))

        def __len__(self):
            return len(self.img_path)

    train_path = glob.glob('../input/train/*.png')
    train_path.sort()
    train_json = json.load(open('../input/train.json'))
    train_label = [train_json[x]['label'] for x in train_json]

    train_loader = torch.utils.data.DataLoader(
            SVHNDataset(train_path, train_label,
                       transforms.Compose([
                           transforms.Resize((64, 128)),
                           transforms.ColorJitter(0.3, 0.3, 0.2),
                           transforms.RandomRotation(5),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])), 
        batch_size=10, # 每批样本个数
        shuffle=False, # 是否打乱顺序
        num_workers=10, # 读取的线程个数
    )

    for data in train_loader:
        break  
  
  

