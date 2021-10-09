import BinaryTreeNode


def createNodes(lines, lineNum , list):
    node = None
    data = lines[int(lineNum)].split(",")
    text = ""
    for i in  range(3,len(data)):
        text = text + data[i]

    a =False
    b =False
    if data[1] in list:
        a= True
    if data[2] in list:
        b= True
    if len(data) < 4:
        print("Not valid file")
        return None
    else:
        if (data[1] =="-" or not a) and (data[2] == "-" or not b):
            node = BinaryTreeNode.BinaryTreeNode(createNodes(lines, data[1],list) if data[1] != "-" else None,
                                                 createNodes(lines, data[2],list) if data[2] != "-" else None, text,
                                                 True if data[1] == "-" else False)
            list[data[0]]=node
        elif (data[1]=="-" or not a ) and (data[2]=="-" or b):
            node = BinaryTreeNode.BinaryTreeNode(createNodes(lines, data[1], list) if data[1] != "-" else None,
                                                 list[data[2]] if data[2] != "-" else None, text,
                                                 True if data[1] == "-" else False)
            list[data[0]]=node
        elif (data[1]=="-" or a) and (data[2]=="-" or not b):
            node = BinaryTreeNode.BinaryTreeNode(list[data[1]] if data[1] != "-" else None,
                                                 createNodes(lines, data[2], list) if data[2] != "-" else None, text,
                                                 True if data[1] == "-" else False)
            list[data[0]]=node
        elif (data[1]=="-" or a) and (data[2]=="-" or b):
            node = BinaryTreeNode.BinaryTreeNode(list[data[1]] if data[1] != "-" else None,
                                                 list[data[2]] if data[2] != "-" else None, text,
                                                 True if data[1] == "-" else False)
            list[data[0]]=node
    return node


def createBinaryTree(file):
    fileReader = open(file, encoding="utf8")
    f = fileReader.read().split("\n")
    list = {}
    tree = createNodes(f, 0,list)
    # for i in list:
    #     print(i , list[i].data)
    return tree
