#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Code base for representing hierarchies/trees for paper Hierarchical Losses for Semantic Segmentation (BMVC 2019 submission).

@author: Bruce Muller (brm512@york.ac.uk) and William Smith (william.smith@york.ac.uk)
"""

#modified to have level too
class node:

    def __init__(self, name):
        self.name = name
        self.children = []
        self.channel = None
        self.level = None

# modified to print level as right most value
def print_tree(node,indent):
	if len(node.children)==0:
		print("\t"*indent + node.name + " " + str(node.channel) + " " + str(node.level))
	else:
		print("\t"*indent + node.name + " " + str(node.level))
		for child in node.children:
			print_tree(child,indent+1)
	return

#returns the number of leaves
def add_channels(node,channel):
    if len(node.children)==0:
        node.channel = channel
        return channel+1
    else:
        for child in node.children:
            channel = add_channels(child,channel)
        return channel
    
def update_channels(node,class_lookup):   # created to map channels in labels to those in tree
    if len(node.children)==0:
	    node.channel = class_lookup[node.channel]
	    return
    else:
 
        for child in node.children:
            update_channels(child,class_lookup)

def find_depth(node):
	if len(node.children)==0:
		return 0
	else:
		maxlen = 0
		for child in node.children:
			maxlen = max(maxlen,find_depth(child))
		return maxlen+1
    
def create_tree_from_textfile(filename):
	# Reads a text file describing a tree
	# Tab indentation indicates parent/child relations
	root = node("Universal class")
	current_depth = 0
	fd = open(filename)
	nodestack = [root]
	# There are three possibilities for each line:
	# 1. Same indent as previous line - new node with same parent as previous line
	# 2. More indented than previous line - new node with parent as previous line
	# 3. Less indented than previous line - number of indents less is how many nodes need popping off the stack to find the parent
	for i,line in enumerate(fd):
		# Create new node with the name taken from the text file after stripping tabs
		newnode = node(line.strip())
		if line.count('\t')==current_depth:
			# Make new node child of node currently at top of stack
			nodestack[-1].children.append(newnode)
			# Make a copy in case next line is indented more
			prevnode = newnode
		elif line.count('\t')==current_depth+1:
			# We have just indented one so the previous line is the parent of this node
			
			# Add node from previous line to top of stack
			nodestack.append(prevnode)
			
			# Make new node child of node currently at top of stack
			nodestack[-1].children.append(newnode)
			
			current_depth += 1

			prevnode = newnode
		elif line.count('\t')<current_depth:
			# Indentation has reduced
			new_depth = line.count('\t')
			# For each reduction in indentation, pop one node off the stack
			while current_depth > new_depth:
				nodestack.pop()
				current_depth -= 1
			# Make new node child of node currently at top of stack
			nodestack[-1].children.append(newnode)
			
			prevnode = newnode
		else:
			raise RuntimeError("Indentation can only increase by one")
	fd.close()
	return root
    
# Needs the depth given as argument. Root not allocated level 
# could get rid of need for depth argument
def add_levels(node,depth):
    #depth = find_depth(node) - 1
    #print(depth)
    
    if len(node.children) == 0:
        node.level = depth - 1
    else:
        for child in node.children:
            if len(child.children) == 0:
                child.level = depth - 1
            else:
                child.level = depth - 1
                add_levels(child,depth-1)
    
# returns list of classes in that node branch                
def getLeafClasses(node, my_list):   
    if len(node.children) == 0:
        my_list.append(node.channel)
        return my_list
    else:
        for child in node.children:
            getLeafClasses(child, my_list)
        return my_list
    

def getLossLevelList(root, level, myList):
    for child in root.children:
        if len(child.children) == 0 or child.level == level:
            myList.append(getLeafClasses(child, []))
        else:
            getLossLevelList(child, level, myList)    
            
            
def getTreeList(node):
    depth = find_depth(node)
    main_list = []
    for level in range(depth):
        level_list = []
        getLossLevelList(node, level, level_list)
        main_list.append(level_list)
    return main_list
#%%    Tested positive on these two trees
         
#root = create_tree_from_textfile("/home/brm512/Pytorch/white_lines/class_hierarchy.txt")
#   
#print_tree(root,0)
#add_channels(root,0)
#print_tree(root,0)
#add_levels(root,3)
#print_tree(root,0)
#

#
#print(getTreeList(root))
#
#root2 = create_tree_from_textfile("/home/brm512/Pytorch/white_lines/class_hierarchy_test1.txt")
#add_channels(root2, 0)
#add_levels(root2, find_depth(root2)-1 )
#
#print(getTreeList(root2))
#
#print_tree(root2,0)
#
#
#whiteline_root = create_tree_from_textfile("/home/brm512/Pytorch/white_lines/class_hierarchy_wl.txt")
#add_channels(whiteline_root, 0)
#add_levels(whiteline_root, find_depth(whiteline_root)-1 )
#
#print(getTreeList(whiteline_root))
#
#print_tree(whiteline_root,0)