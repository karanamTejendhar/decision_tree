import BinaryTree
from forNewExample import NewQuery,Information,AllPaths
import pickle

model = pickle.load(open('model.sav','rb'))

print("Enter the name of the object you want to ask about")
objectName = input().lower()
path = "./trees/"+objectName+".classification"

n = BinaryTree.createBinaryTree(path)
if n is None:
    print("No tree defined")



while True:
    if n is None:
        break
    print("\n\n")
    print("Enter 1 for new Example \nEnter 2 for information about object\nEnter 3 for all necessary paths")
    QuestionType = input()
    if QuestionType == "1":
        NewQuery(objectName)
    elif QuestionType == "2":
        Information(objectName,model)
    elif QuestionType == "3":
        AllPaths(objectName)
    else:
        print("please enter a proper question type")

# print("Your answer is a", n.data if n is not None else "No answer in tree")
