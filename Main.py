# noinspection PyUnresolvedReferences
from collections import namedtuple
import math


#  This class simply takes the file provided and puts it into a manageable form
class DataStucture:
    treeNode = namedtuple("treeNode", "value, PositiveVNegative, gain, left, right")
    FeatureValue = namedtuple("FeatureValue", "feature, value")
    positive = 'e'
    empty = None

    def __init__(self, fileName):
        self.featureSet = set()
        self.nodes = []
        openedFile = None
        try:
            openedFile = open(fileName, 'r')
        except FileNotFoundError:
            print("Error: Please rerun and enter valid csv file name.")
            exit(0)
        for eachLine in openedFile:
            eachLine = eachLine.strip()
            split = eachLine.split(',')  # takes commas out of csv data
            n = 1
            while n < len(split):
                self.featureSet.add(DataStucture.FeatureValue(n, split[n]))  # add all features to feature set
                n += 1
            self.nodes.append(split)


def calculateEntropy(struct):
    # total of positive and negative items initially 0
    totalPositive = 0
    totalNegative = 0
    # total number of positive and negative items
    for dp in struct:
        if dp[0] is DataStucture.positive:
            totalPositive += 1
        else:
            totalNegative += 1
    # probablities
    positiveProbability = (totalPositive + 1) / (len(struct) + 2)
    negativeProbability = (totalNegative + 1) / (len(struct) + 2)
    # calulate entropy based on entropy equation
    entropy = -positiveProbability * math.log(positiveProbability, 2) - negativeProbability * math.log(
        negativeProbability, 2)
    return entropy


def infoGain(struct, f):
    leftTree = 0
    rightTree = 0
    leftBranch = []
    rightBranch = []

    for item in struct:  # looking at each item for the features
        if item[f[0]] is f[1]:
            leftBranch.append(item)
            leftTree += 1
        else:
            rightBranch.append(item)
            rightTree += 1
    # probabilites for each branch
    ltreeProb = leftTree / len(struct)
    rtreeProb = rightTree / len(struct)
    # calulate info gain from equation
    infogain = calculateEntropy(struct) - rtreeProb * calculateEntropy(rightBranch) - ltreeProb * calculateEntropy(leftBranch)
    return infogain


def calculateAccuracy(treeStruct, struct):
    correct = 0
    for dp in struct:
        feature = classifyFeature(treeStruct, dp)
        comp = (dp[0] == DataStucture.positive)
        if feature == comp:
            correct += 1
    accuracy = float(correct) / len(struct)
    return accuracy


# noinspection PyTypeChecker
def classifyFeature(tree, dp):
    # Recursively classify the features in the tree
    if tree.left is None and tree.right is None:
        return tree.PositiveVNegative[0] > tree.PositiveVNegative[1]
    elif dp[tree.value.feature] == tree.value.value:
        # noinspection PyTypeChecker
        return classifyFeature(tree.left, dp)
    else:
        return classifyFeature(tree.right, dp)


def prettyPrintTree(node):
    # Recursively pretty print the final tree
    print(f"Node: {node.value} \n Statistics: Number of positive vs "
          f"Negative: {node.PositiveVNegative}, Gain of Node: {node.gain}")
    if node.left is not None:
        prettyPrintTree(node.left)
    if node.right is not None:
        prettyPrintTree(node.right)


def id3(struct, classes, minGain):
    value = DataStucture.FeatureValue
    positiveNegative = [0, 0]
    base = [0, 0]
    gain = 0.0
    # Lists representing the left and right nodes
    left = []
    right = []
    index = 0
    # Search features and save feature with highest gain
    for c in classes:
        igain = infoGain(struct, c)
        if gain < igain:
            gain = igain
            value = c

    # check if current gain is less than the supplied min gain
    if gain <= minGain:
        for data in struct:
            if data[0] is DataStucture.positive:
                index = 0
            else:
                index = 1
            base[index] += 1
        tree = DataStucture.treeNode(value, base, gain, DataStucture.empty, DataStucture.empty)
        return tree

    for data in struct:
        if data[value[0]] is value[1]:
            left.append(data)
            index = 0
        else:
            right.append(data)
            index = 1
        positiveNegative[index] += 1

    id3Left = id3(left, classes, minGain)
    id3Right = id3(right, classes, minGain)

    tree = DataStucture.treeNode(value, positiveNegative, gain, id3Left,
                                 id3Right)
    return tree


if __name__ == "__main__":

    ans = None
    while ans != 0:
        arg1, arg2, arg3 = None, None, None

        ans = input("Would you like to provide training and testing data (1), use "
                    "the preset filenames (test.csv and train.csv, make sure these are in the same directory)(2) "
                    "or quit? (0) ")
        try:
            ans = int(ans)
        except ValueError:
            print("Error: Input should be an int")
            exit()

        if ans == 1:
            arg1 = input("Enter training data csv file: ")
            arg2 = input("Enter testing data csv file: ")
            arg3 = input("Enter minimum gain for test: ")
            try:
                arg3 = float(arg3)
            except ValueError:
                print("Error: Min gain should be an number")
                exit(0)
            if arg3 < 0 or arg3 >= 1:
                print("Error: Value for min gain should be between 0 and 1")
                exit(0)
        elif ans == 2:
            arg2 = "test.csv"
            arg1 = "train.csv"
            arg3 = 0
        elif ans == 0:
            exit(1)
        else:
            print("Error: Invalid input. Please rerun and try again")
            exit(0)

        if arg1 is not None and arg2 is not None and arg3 is not None:
            train = DataStucture(arg1)
            test = DataStucture(arg2)
            print("Running ID3 algorithm...")
            decisionTree = id3(train.nodes, train.featureSet, arg3)
            print("Root ", end='')
            prettyPrintTree(decisionTree)

            acc = calculateAccuracy(decisionTree, test.nodes)
            print("Accuracy is", acc)
        else:
            print("Error occurred, try again")
