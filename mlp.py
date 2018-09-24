from torch import nn

from initialization import initialize_weights


class MLP(nn.Module):

	def __init__(self,
			input_dimension,
			output_dimension,
			hidden_dimension=64,
			num_layers=1,
			hidden_activation=nn.ReLU,
			output_activation=nn.Softmax):

		super(MLP, self).__init__()
		self.layers = nn.ModuleList()
		self.hidden_activation = hidden_activation()
		self.output_activation = output_activation()
		for i in range(num_layers + 1):
			if i == 0:
				self.layers.append(nn.Linear(input_dimension, hidden_dimension, bias=True))
			elif i == num_layers:
				self.layers.append(nn.Linear(hidden_dimension, output_dimension, bias=True))
			else:
				self.layers.append(nn.Linear(hidden_dimension, hidden_dimension, bias=True))
		self.initialize_modules()

	def initialize_modules(self):
		for module in self.modules():
			module.apply(initialize_weights)

	def forward(self, x, return_logits=True):
		for i, layer in enumerate(self.layers):
			if i == len(self.layers) - 1:
				if return_logits:
					x = layer(x)
				else:
					x = self.output_activation(layer(x))
			else:
				x = self.hidden_activation(layer(x))
		return x