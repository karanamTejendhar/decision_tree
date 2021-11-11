import BinaryTree
import os
from cleanSentences import get_cleaned_sentences
import pickle

def NewQuery(objectName):
    path = "./trees/" + objectName + ".classification"
    n = BinaryTree.createBinaryTree(path)
    print("Answer question using yes/no or y/n")
    while n is None or n.terminating is False:
        if n is None:
            print("No answer in tree defined")
            break
        answer = input(n.data.replace("\\n", "\n") + " ").lower()

        if answer in ["y", "yes"]:
            n = n.left
        elif answer in ["n", "no"]:
            n = n.right
        else:
            print("Not a valid answer! Only yes and no are allowed!")

    print("Your answer is a", n.data if n is not None else "No answer in tree")


def Information(objectName,model):
    path = "./trees/" + objectName + ".classification"
    n = BinaryTree.createBinaryTree(path)
    final_list = []
    templist = []

    question = input("Please enter the question\n")

    def traversal(n, templist, final_list):
        if n is None:
            print("No answer in defined tree")
            return
        if n.left is not None:
            templist.append(n.data)
            traversal(n.left, templist, final_list)
            templist.pop()
        if n.right is not None:
            traversal(n.right, templist, final_list)

        if n.data.lower().lstrip().rstrip() == objectName.lstrip().rstrip() and n.left is None and n.right is None:
            templist_clone = templist[:]  # for making it cloning
            final_list.append(templist_clone)

    if n is None or n.terminating is False:
        traversal(n, templist, final_list)
    # print(final_list)

    for alist in final_list:
        sentences= alist
        get_cleaned_sentences(sentences,question,model,stopwords= False)

def AllPaths(objectName):
    path = "./trees/" + objectName + ".classification"
    n = BinaryTree.createBinaryTree(path)
    final_list = []
    templist = []


    def traversal(n, templist, final_list):
        if n is None:
            print("No answer in defined tree")
            return
        if n.left is not None:
            templist.append(n.data)
            traversal(n.left, templist, final_list)
            templist.pop()
        if n.right is not None:
            traversal(n.right, templist, final_list)

        if n.data.lower().lstrip().rstrip() == objectName.lstrip().rstrip() and n.left is None and n.right is None:
            templist_clone = templist[:]  # for making it cloning
            final_list.append(templist_clone)

    if n is None or n.terminating is False:
        traversal(n, templist, final_list)

    for one in final_list:
        for x in one:
            print(x, end=" --> ")
        print("done")