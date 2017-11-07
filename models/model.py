import os
import torch
import torch.nn as nn
import torchvision.models as models

class bkpmodel(object):
	"""docstring for bkpmodel"""
	def __init__(self, arg):
		super(bkpmodel, self).__init__()
		self.arg = arg
		self.basemodel = models.alexnet(pretrained=True)
		# for param in self.basemodel.parameters():
		# 	param.requires_grad = False
		self.basemodel.classifier._modules['6'] = nn.Linear(in_features = 4096, out_features=256)
		self.confusionmodel = nn.Sequential(
		        nn.Dropout(0.5),
		        nn.Linear(in_features=256,out_features=1024),
		        nn.Relu(inplace=True),
		        nn.Dropout(0.5),
		        nn.Linear(in_features = 1024, out_features = 1024),
		        nn.Relu(inplace=True),
		        nn.Dropout(0.5),
		        nn.Linear(in_features=1024,out_features=2)
		                                    )
		self.classify_model = nn.Linear(in_features=256, out_features=31 )

		if self.isTrain:
			self.criterion_class = nn.CrossEntropyLoss()
			self.criterion_domain = nn.CrossEntropyLoss()
			self.optimizer_f_c = torch.optim.Adam(
			    [self.basemodel.parameters(),self.classify_model.parameters(), self.confusionmodel.parameters()],lr=self.lr,momentum=0.9                                  )
			# self.optimizer_d = torch.optim.Adam(
			#     self.confusionmodel.parameters(), lr = self.lr, momentum = 0.9                                )
			

	def rgl_hook(grad):
		grad_clone = grad.clone()
		grad_clone = -lamb*grad_clone
		return grad_clone

	def set_input(self, input):
		self.input = x
	
	def forward(self):
		f = basemodel(self.input)
		N,_ = size(f)
		soure_f = f[:N/2,:,:,:]
		self.predict_class_label= self.classify_model(soure_f)
		f_con = f
		f_con.register_hook(self.rgl_hook)
		self.predict_domain_label = self.confusionmodel(f_con)

	def backward():
		class_loss = self.criterion_class(self.predict_class_label,self.class_label)
		confusion_loss = self.criterion_domain(self.predict_domain_label, self.domain_label)

		loss = class_loss + confusion_loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step() 
		
	def optimize_parameters():
		self.forward()
		self.backward()

		self.optimize_f_c()

		pass
	def save_network():
		pass

	def load_network():
		pass

	def update_learning_rate():
		pass