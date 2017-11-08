import os
import torch
import torch.nn as nn
import torchvision.models as models

class bkpmodel(nn.Module):
	"""docstring for bkpmodel"""
	def __init__(self):
		super(bkpmodel, self).__init__()
		# self.arg = arg
		self.basemodel = models.alexnet(pretrained=True)
		# for param in self.basemodel.parameters():
		# 	param.requires_grad = False
		self.basemodel.classifier._modules['6'] = nn.Linear(in_features = 4096, out_features=256)
		self.confusionmodel = nn.Sequential(
		        nn.Dropout(0.5),
		        nn.Linear(in_features=256,out_features=1024),
		        nn.ReLU(inplace=True),
		        nn.Dropout(0.5),
		        nn.Linear(in_features = 1024, out_features = 1024),
		        nn.ReLU(inplace=True),
		        nn.Dropout(0.5),
		        nn.Linear(in_features=1024,out_features=2)
		                                    )
		self.classify_model = nn.Linear(in_features=256, out_features=31 )
	# def train(self):
	# 	self.criterion_class = nn.CrossEntropyLoss()
	# 	self.criterion_domain = nn.CrossEntropyLoss()
	# 	self.optimizer_f_c = torch.optim.Adam(
	# 	    [self.basemodel.parameters(),self.classify_model.parameters(), self.confusionmodel.parameters()],lr=self.lr,momentum=0.9                                  )
	# 	# self.optimizer_d = torch.optim.Adam(
	# 	#     self.confusionmodel.parameters(), lr = self.lr, momentum = 0.9                                )
			
	def rgl_hook(grad):
		grad_clone = grad.clone()
		grad_clone = -lamb*grad_clone
		return grad_clone

	# def set_input(self, input):
	# 	self.input = x
	
	def forward(self,x):
		f = self.basemodel(x)
		N,_ = len(f)
		soure_f = f[:N/2,:,:,:]
		predict_class_label= self.classify_model(soure_f)
		f_con = f
		h = f_con.register_hook(self.rgl_hook)
		predict_domain_label = self.confusionmodel(f_con)
		return predict_class_label, predict_domain_label

	# def backward():
	# 	class_loss = self.criterion_class(self.predict_class_label,self.class_label)
	# 	confusion_loss = self.criterion_domain(self.predict_domain_label, self.domain_label)

	# 	loss = class_loss + confusion_loss
	# 	self.optimizer.zero_grad()
	# 	loss.backward()
	# 	self.optimizer.step() 
		
	# def optimize_parameters():
	# 	self.forward()
	# 	self.backward()

	# 	self.optimize_f_c()

	# 	pass
	# def save_network():
	# 	pass

	# def load_network():
	# 	pass

	# def update_learning_rate():
	# 	pass