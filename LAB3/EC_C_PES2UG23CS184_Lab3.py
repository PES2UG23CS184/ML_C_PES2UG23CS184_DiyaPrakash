import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
	"""
	Calculate the entropy of the entire dataset.
	Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i

	Args:
		tensor (torch.Tensor): Input dataset as a tensor, where the last column is the target.

	Returns:
		float: Entropy of the dataset.
	"""
	# Get the last column (target)
	target = tensor[:, -1]
	values, counts = torch.unique(target, return_counts=True)
	probs = counts.float() / target.size(0)
	entropy = -(probs * torch.log2(probs)).sum().item()
	return round(entropy, 5)


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
	"""
	Calculate the average information (weighted entropy) of an attribute.
	Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.

	Args:
		tensor (torch.Tensor): Input dataset as a tensor.
		attribute (int): Index of the attribute column.

	Returns:
		float: Average information of the attribute.
	"""
	# Get unique values for the attribute
	attr_col = tensor[:, attribute]
	total = tensor.size(0)
	avg_info = 0.0
	for v in torch.unique(attr_col):
		subset = tensor[attr_col == v]
		if subset.size(0) == 0:
			continue
		subset_entropy = get_entropy_of_dataset(subset)
		avg_info += (subset.size(0) / total) * subset_entropy
	return round(avg_info, 5)


def get_information_gain(tensor: torch.Tensor, attribute: int):
	"""
	Calculate Information Gain for an attribute.
	Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)

	Args:
		tensor (torch.Tensor): Input dataset as a tensor.
		attribute (int): Index of the attribute column.

	Returns:
		float: Information gain for the attribute (rounded to 4 decimals).
	"""
	entropy = get_entropy_of_dataset(tensor)
	avg_info = get_avg_info_of_attribute(tensor, attribute)
	info_gain = entropy - avg_info
	return round(info_gain, 5)


def get_selected_attribute(tensor: torch.Tensor):
	"""
	Select the best attribute based on highest information gain.
	Returns a tuple with:
	1. Dictionary mapping attribute indices to their information gains
	2. Index of the attribute with highest information gain
	Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)

	Args:
		tensor (torch.Tensor): Input dataset as a tensor.

	Returns:
		tuple: (dict of attribute:index -> information gain, index of best attribute)
	"""
	n_attributes = tensor.size(1) - 1  # Exclude target
	info_gains = {}
	for i in range(n_attributes):
		info_gains[i] = get_information_gain(tensor, i)
	selected = max(info_gains, key=info_gains.get)
	return info_gains, selected
