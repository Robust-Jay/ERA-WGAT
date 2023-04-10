import os
import numpy as np
from sklearn.model_selection import train_test_split

train = [i for i in range(8)]
test = [8, 9]

train_list = []
root = "../../../datasets/Mayo/full_dose"
for item in train:
	files = os.listdir(f"{root}/{item + 1}")
	temp = [i for i in range(len(files))]
	temp = temp[4:-4]  # max_depth = 9

	train_list.extend([f"{item + 1}/{t}.npy" for t in temp])
	# train_list.extend([f"{item + 1}/{value}" for value in os.listdir(f"{root}/{item + 1}")])

test_list = []
for item in test:
	files = os.listdir(f"{root}/{item + 1}")
	temp = [i for i in range(len(files))]
	temp = temp[4:-4]  # max_depth = 9
	test_list.extend([f"{item + 1}/{t}.npy" for t in temp])

	# test_list.extend([f"{item + 1}/{value}" for value in os.listdir(f"{root}/{item + 1}")])

print(len(train_list))
print(len(test_list))

np.save("split_datasets/train.npy", train_list, allow_pickle=False)
np.save("split_datasets/test_tol.npy", test_list, allow_pickle=False)



if __name__ == '__main__':
	# train_set = np.load("train_tol.npy", allow_pickle=False)
	# train, val = train_test_split(train_set, test_size=0.1)
	# np.save("train.npy", train, allow_pickle=False)
	# np.save("val.npy", val, allow_pickle=False)
	# print()

	test_tol = np.load("split_datasets/test_tol.npy", allow_pickle=False)
	test, val = train_test_split(test_tol, test_size=0.2, random_state=32)
	np.save("split_datasets/test.npy", test, allow_pickle=False)
	np.save("split_datasets/val.npy", val, allow_pickle=False)
	print()
