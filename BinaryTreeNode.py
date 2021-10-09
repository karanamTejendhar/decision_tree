class BinaryTreeNode:

	left = ""
	right = ""
	data = ""
	terminating = False
	def __init__(self, left, right, data, terminating):
		self.left = left
		self.right = right
		self.data = data
		self.terminating = terminating
