from random import seed
from random import randrange
from csv import reader
from math import sqrt
from PIL import Image, ImageTk
import PySimpleGUI as sg

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
 
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)
 
# Test the kNN on the Iris Flowers dataset
seed(1)
filename = 'StarData\Stars.csv'
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
 
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)
 
# Test the kNN on the Iris Flowers dataset
seed(1)
filename = 'StarData\Stars.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
#str_column_to_int(dataset, len(dataset[0])-1)

# evaluate algorithm
n_folds = 5
num_neighbors = 5
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
#str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
num_neighbors = 5

scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

sg.theme('DarkPurple1')
layout1 = [
    [sg.Text('Predicting Star Types')],
	[sg.Text('Enter data to determine the type of star')],
    [sg.Submit(), sg.Cancel()]
]

window = sg.Window('Star Prediction', layout1, element_justification='c', resizable=True, finalize=True)
window.set_min_size((300,100))
event = window.read()
window.close()

layout2 = [
	[sg.Text('Please Enter Star Temperature', size =(25, 1)), sg.InputText()],
	[sg.Text('Please Enter Star Luminosity', size =(25, 1)), sg.InputText()],
	[sg.Text('Please Enter Star Radius', size =(25, 1)), sg.InputText()],
	[sg.Text('Please Enter Star Magnitude', size =(25, 1)), sg.InputText()],
	[sg.Submit(), sg.Cancel()]
]

window = sg.Window('Input Values', layout2, element_justification='c', resizable=True, finalize=True)
window.set_min_size((300,100))
event, values1 = window.read()
window.close()

Temperature=values1[0]
Luminosity=values1[1]
Radius=values1[2]
Magnitude=values1[3]

layout3=[
	[sg.Text('Please Pick Star Color')],
	[sg.Text('Blue, White, Yellow, Orange, or Red?'),sg.Combo(["Blue","White","Yellow","Orange","Red"])],
	[sg.Submit(), sg.Cancel()]
	]

window = sg.Window('Input Values', layout3, element_justification='c', resizable=True, finalize=True)
window.set_min_size((300,100))
event, values2 = window.read()
window.close()

if values2[0] == "Blue":
	starColor=0
elif values2[0]=="White":
	starColor=1
elif values2[0]=="Yellow":
	starColor=2
elif values2[0]=="Orange":
	starColor=3
elif values2[0]=="Red":
	starColor=4

layout4=[
	[sg.Text('Please Enter Star Spectral Class')],
	[sg.Text('O, B, A, F, G, K, or M?'), sg.Combo(["O","B","A","F","G","K","M"])],
	[sg.Submit(), sg.Cancel()]
	]
window = sg.Window('Input Values', layout4, element_justification='c', resizable=True, finalize=True)
window.set_min_size((300,100))
event, values3 = window.read()
window.close()

if values3[0]=="O":
	spectralClass=0
elif values3[0]=="B":
	spectralClass=1
elif values3[0]=="A":
	spectralClass=2
elif values3[0]=="F":
	spectralClass=3
elif values3[0]=="G":
	spectralClass=4
elif values3[0]=="K":
	spectralClass=5
elif values3[0]=="M":
	spectralClass=6

row=[float(Temperature),float(Luminosity),float(Radius),float(Magnitude),float(starColor),float(spectralClass)]

label=predict_classification(dataset, row, num_neighbors)

size=(400,300)
RedDwarf= Image.open('StarData\RedDwarf.jpg')
RedDwarf=RedDwarf.resize(size, resample=Image.BICUBIC)
BrownDwarf=Image.open('StarData\BrownDwarf.png')
BrownDwarf=BrownDwarf.resize(size, resample=Image.BICUBIC)
WhiteDwarf=Image.open('StarData\WhiteDwarf.png')
WhiteDwarf=WhiteDwarf.resize(size, resample=Image.BICUBIC)
MainSequence=Image.open('StarData\main-sequence-star.png')
MainSequence=MainSequence.resize(size, resample=Image.BICUBIC)
SuperGiant=Image.open('StarData\SuperGiant.png')
SuperGiant=SuperGiant.resize(size, resample=Image.BICUBIC)
HyperGiant=Image.open('StarData\HyperGiant.png')
HyperGiant=HyperGiant.resize(size, resample=Image.BICUBIC)


if label=='0':
	layout5=[[sg.Text("The Star is a Red Dwarf")],
	[sg.Image(key='-IMAGE-')],
	[sg.Submit(), sg.Cancel()]]
	Display=RedDwarf
elif label=='1':
	layout5=[[sg.Text("The Star is a Brown Dwarf")],
	[sg.Image(size=(100,100),key='-IMAGE-')],
	[sg.Submit(), sg.Cancel()]]
	Display=BrownDwarf
elif label=='2':
	layout5=[[sg.Text("The Star is a White Dwarf")],
	[sg.Image(size=(100,100),key='-IMAGE-')],
	[sg.Submit(), sg.Cancel()]]
	Display=WhiteDwarf
elif label=='3':
	layout5=[[sg.Text("The Star is a Main Sequence")],
	[sg.Image(size=(100,100),key='-IMAGE-')],
	[sg.Submit(), sg.Cancel()]]
	Display=MainSequence
elif label=='4':
	layout5=[[sg.Text("The Star is a Super Giant")],
	[sg.Image(size=(100,100),key='-IMAGE-')],
	[sg.Submit(), sg.Cancel()]]
	Display=SuperGiant
elif label=='5':
	layout5=[[sg.Text("The Star is a Hyper Giant")],
	[sg.Image(size=(100,100),key='-IMAGE-')],
	[sg.Submit(), sg.Cancel()]]
	Display=HyperGiant

window = sg.Window("What is the Star Type?", layout5, element_justification='c', resizable=True, finalize=True)
window.set_min_size((300,100))
image = ImageTk.PhotoImage(image=Display)
window['-IMAGE-'].update(data=image)
event = window.read()
window.close()