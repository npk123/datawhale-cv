# Task4 模型训练与验证

# 构造验证集
深度学习模型在不断的训练过程中易造成过拟合，模型对训练集上的数据有很好的解释能力，但在测试集上表现较差，即学习的模型泛化能力差。
随着模型复杂度和模型训练轮数的增加，CNN模型在训练集上的误差会降低，但在测试集上的误差会逐渐降低，然后逐渐升高，因此一味追求训练集精度并非最终目的。

导致模型过拟合的情况有很多种原因，其中最为常见的情况是模型复杂度（Model Complexity ）太高，导致模型学习到了训练数据的方方面面，学习到了一些细枝末节的规律。（用正则化矫正模型复杂度）
解决上述问题最好的解决方法：构建一个与测试集尽可能分布一致的样本集（可称为验证集），在训练过程中不断验证模型在验证集上的精度，并以此控制模型的训练

# 模型评估方式

* 训练集（Train Set）：模型用于训练和调整模型参数
* 验证集（Validation Set）：用来验证模型精度和调整模型超参数
* 测试集（Test Set）：验证模型的泛化能力

# 模型训练与验证
* 构造训练集和验证集；
* 每轮进行训练和验证，并根据最优验证集精度保存模型。

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=10, 
		shuffle=True, 
		num_workers=10, 
	)

	val_loader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=10, 
		shuffle=False, 
		num_workers=10, 
	)

	model = SVHN_Model1()
	criterion = nn.CrossEntropyLoss (size_average=False)   # CrossEntropy是LogSoftmax和NLLLoss的结合，适用于多分类问题 https://mfy.world/deep-learning/pytorch/pytorchnotes-lossfunc/
	optimizer = torch.optim.Adam(model.parameters(), 0.001)  # 优化算法/ Adam相当于 RMSprop + Momentum https://www.cnblogs.com/guoyaohua/p/8542554.html
	best_loss = 1000.0

	for epoch in range(20):
		print('Epoch: ', epoch)

		train(train_loader, model, criterion, optimizer, epoch)
		val_loss = validate(val_loader, model, criterion)

		# 记录下验证集精度
		if val_loss < best_loss:
			best_loss = val_loss
			torch.save(model.state_dict(), './model.pt')
			
	def train(train_loader, model, criterion, optimizer, epoch):
		# 切换模型为训练模式
		model.train()

		for i, (input, target) in enumerate(train_loader):
			c0, c1, c2, c3, c4, c5 = model(data[0])
			loss = criterion(c0, data[1][:, 0]) + \
					criterion(c1, data[1][:, 1]) + \
					criterion(c2, data[1][:, 2]) + \
					criterion(c3, data[1][:, 3]) + \
					criterion(c4, data[1][:, 4]) + \
					criterion(c5, data[1][:, 5])
			loss /= 6
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
	def validate(val_loader, model, criterion):
		# 切换模型为预测模型
		model.eval()
		val_loss = []

		# 不记录模型梯度信息
		with torch.no_grad():
			for i, (input, target) in enumerate(val_loader):
				c0, c1, c2, c3, c4, c5 = model(data[0])
				loss = criterion(c0, data[1][:, 0]) + \
						criterion(c1, data[1][:, 1]) + \
						criterion(c2, data[1][:, 2]) + \
						criterion(c3, data[1][:, 3]) + \
						criterion(c4, data[1][:, 4]) + \
						criterion(c5, data[1][:, 5])
				loss /= 6
				val_loss.append(loss.item())
		return np.mean(val_loss)
		
# 模型的保存与加载
比较常见的做法是保存和加载模型参数：
	torch.save(model_object.state_dict(),'model.pt')
加载模型参数
	model.load_state_dict(torch.load('model.pt'))

这里对深度学习的训练技巧推荐的阅读链接：
http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
http://karpathy.github.io/2019/04/25/recipe/
https://www.cnblogs.com/whiteBear/p/12986528.html
