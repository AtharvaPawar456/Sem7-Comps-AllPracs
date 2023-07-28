# Python code for implemementing Merkle Tree
from typing import List
import hashlib

class Node:
	def __init__(self, left, right, value: str, content, is_copied=False) -> None:
		self.left: Node = left
		self.right: Node = right
		self.value = value
		self.content = content
		self.is_copied = is_copied
		
	@staticmethod
	def hash(val: str) -> str:
		return hashlib.sha256(val.encode('utf-8')).hexdigest()

	def __str__(self):
		return (str(self.value))

	def copy(self):
		"""
		class copy function
		"""
		return Node(self.left, self.right, self.value, self.content, True)
	
class MerkleTree:
	def __init__(self, values: List[str]) -> None:
		self.__buildTree(values)

	def __buildTree(self, values: List[str]) -> None:

		leaves: List[Node] = [Node(None, None, Node.hash(e), e) for e in values]
		if len(leaves) % 2 == 1:
			leaves.append(leaves[-1].copy()) # duplicate last elem if odd number of elements
		self.root: Node = self.__buildTreeRec(leaves)

	def __buildTreeRec(self, nodes: List[Node]) -> Node:
		if len(nodes) % 2 == 1:
			nodes.append(nodes[-1].copy()) # duplicate last elem if odd number of elements
		half: int = len(nodes) // 2

		if len(nodes) == 2:
			return Node(nodes[0], nodes[1], Node.hash(nodes[0].value + nodes[1].value), nodes[0].content+"+"+nodes[1].content)

		left: Node = self.__buildTreeRec(nodes[:half])
		right: Node = self.__buildTreeRec(nodes[half:])
		value: str = Node.hash(left.value + right.value)
		content: str = f'{left.content}+{right.content}'
		return Node(left, right, value, content)

	def printTree(self) -> None:
		self.__printTreeRec(self.root)
		
	def __printTreeRec(self, node: Node) -> None:
		if node != None:
			if node.left != None:
				print("Left: "+str(node.left))
				print("Right: "+str(node.right))
			else:
				print("Input")
				
			if node.is_copied:
				print('(Padding)')
			print("Value: "+str(node.value))
			print("Content: "+str(node.content))
			print("")
			self.__printTreeRec(node.left)
			self.__printTreeRec(node.right)

	def getRootHash(self) -> str:
	    return self.root.value

def mixmerkletree() -> List[str]:
    
	## testcase:

	# testcase - 1
    elems = ["GeeksforGeeks", "Computer", "Good", "Morning"]
	
	# testcase - 2
    # elems = ["Geeksfor", "Geeks", "Computer", "Science","Good", "Morning", "Block", "Chain"]
    
    # as there are odd number of inputs, the last input is repeated
    output = []

    output.append("Inputs: ")
    output.append(" | ".join(elems))
    output.append("")

    mtree = MerkleTree(elems)
    output.append("Root Hash: " + mtree.getRootHash())
    output.append("")
    
    # Modified version of printTree to add lines to the output list
    def printTreeRec(node: Node, depth: int) -> None:
        if node is not None:
            if node.left is not None:
                output.append("  	" * depth + "Left: " + str(node.left))
                output.append("  	" * depth + "Right: " + str(node.right))
            else:
                output.append("  	" * depth + "Input")

            if node.is_copied:
                output.append("  	" * depth + '(Padding)')

            output.append("  	" * depth + "Value: " + str(node.value))
            output.append("  	" * depth + "Content: " + str(node.content))
            output.append("")
            printTreeRec(node.left, depth + 1)
            printTreeRec(node.right, depth + 1)

    printTreeRec(mtree.root, 0)
    return output


output_list = mixmerkletree()
# print(output_list)

temp = 0
for line in output_list:
    print(line)
    
"""
##################
Terminal Output:
##################

Inputs: 
GeeksforGeeks | Computer | Good | Morning

Root Hash: bc6eaec7209f476f6212612b772a3e474a41e3dae28cd740523b39516a04e954

Left: 20999f1bb1e4df7bc51188f9de409c31cf67e83f3ae21d47aca9a201a710c7b1
Right: f89b60d5fbe4181598f2e9efab1375d55e73dd1351017384dc8a59c57d625e94
Value: bc6eaec7209f476f6212612b772a3e474a41e3dae28cd740523b39516a04e954
Content: GeeksforGeeks+Computer+Good+Morning

        Left: f6071725e7ddeb434fb6b32b8ec4a2b14dd7db0d785347b2fb48f9975126178f
        Right: 76ed42d22129dc354362704eb4b54208041b68736f976932aada43bc0035f7c0
        Value: 20999f1bb1e4df7bc51188f9de409c31cf67e83f3ae21d47aca9a201a710c7b1
        Content: GeeksforGeeks+Computer

                Input
                Value: f6071725e7ddeb434fb6b32b8ec4a2b14dd7db0d785347b2fb48f9975126178f
                Content: GeeksforGeeks

                Input
                Value: 76ed42d22129dc354362704eb4b54208041b68736f976932aada43bc0035f7c0
                Content: Computer

        Left: c939327ca16dcf97ca32521d8b834bf1de16573d21deda3bb2a337cf403787a6
        Right: e9376a281aac57bb78e2c769584e5eda9bb93699d299c3a42adc46b7b8e1ccd6
        Value: f89b60d5fbe4181598f2e9efab1375d55e73dd1351017384dc8a59c57d625e94
        Content: Good+Morning

                Input
                Value: c939327ca16dcf97ca32521d8b834bf1de16573d21deda3bb2a337cf403787a6
                Content: Good

                Input
                Value: e9376a281aac57bb78e2c769584e5eda9bb93699d299c3a42adc46b7b8e1ccd6
                Content: Morning
"""