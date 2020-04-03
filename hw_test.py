word = input("hello world")
word = word.strip()
len_last_word = len(word) - word.rindex(" ") - 1
print(len_last_word)

data_test = input()
m = int(data_test.split(" ")[0])
n = int(data_test.split(" ")[1])
# count = 0
count = 1
n_to_2 = int((n - 1) / 2)
count = count + n_to_2
for i in range(3, m + 1):
	count = count + n - i
print(count)

nm = list(input().split(" "))
n = int(nm[1])
m = int(nm[0])

dp = [[0 for i in range(m + 1)] for j in range(n + 1)]
dp[0][0] = 0

for i in range(1, m + 1):
	dp[0][i] = 1

for i in range(1, n + 1):
	for j in range(1, m + 1):
		if i >= j:
			dp[i][j] = dp[i][j - 1] + dp[i - j][j]
		else:
			dp[i][j] = dp[i][j - 1]
		print(dp)
print(str(dp[n][m]))

while True:
	try:
		num_count = int(input())
		input_set = set({})
		for i in range(num_count):
			num_input = int(input())
			input_set.add(num_input)
		num_sort = list(input_set)
		num_sort.sort()
		for num in num_sort:
			print(num)
	except:
		break


def output_str(str_1):
	len_add = 0
	if ((len(str_1) % 8) == 0):
		print(str_1)
	else:
		len_add = (int(len(str_1) / 8) + 1) * 8 - len(str_1)
	for i in range(len_add):
		str_1 = str_1 + "0"

	for j in range(0, int(len(str_1) / 8)):
		print(str_1[j * 8:j * 8 + 8])


str_1 = input()
srr_2 = input()
output_str(str_1)
output_str(srr_2)

str_1.replace(" ", "%20")


def get_input():
	a = []
	b = []
	while True:
		try:
			input_str = input()
			s = int(input_str.split(",")[0])
			f = int(input_str.split(",")[1])
			if (s == 0 and f == 0):
				break
			else:
				a.append([s, f, f - s])
				b.append(f - s)
		except:
			break
	return a, b


def get_result(data_input, data_diff):
	result = []
	temp = data_diff
	for i in range(len(data_diff)):
		row = temp.index(max(data_diff))
		print(str(i) + ":" + str(row))
		print(data_input[i])
		if (len(result) == 0):
			result = [data_input[row]]
		for i in range(len(result)):
			# print(data_input[row])
			if (data_input[row][0] > result[i][0] or data_input[row][1] < result[i][0]):
				# print(data_input[row])
				result.append(data_input[row])

	# data_diff.remove(max(data_diff))
	return result


def unfinish(data_input, result):
	for j in range(len(data_input)):
		max_diff = 0
		print(data_input)
		print(result)
		if (len(data_input) > 0):
			for i in range(len(data_input)):
				if (data_input[i][2] > max_diff):
					max_diff = data_input[i][2]
					print(max_diff)
		else:
			break
		for i in range(len(data_input)):
			if (max_diff == data_input[i][2]):
				# print(str(data_input[i][0]) + "," + str(data_input[i][1]))

				if (len(result) == 0):
					print(str(data_input[i][0]) + "," + str(data_input[i][1]))
					result.append(data_input[i])
					data_input.remove(data_input[i])
				else:
					flag = 1
					for k in range(len(result)):
						if (data_input[i][1] < result[k][0] or data_input[i][0] > result[k][1]):
							flag = 1
						else:
							flag = 0
							print(data_input[i])
							data_input.remove(data_input[i])
							break
					if (flag == 1):
						print(str(data_input[i][0]) + "," + str(data_input[i][1]))
						result.append(data_input[i])
						data_input.remove(data_input[i])
				# print(result)
				continue


data_input, data_diff = get_input()
data_input = [[8, 10, 2], [9, 11, 2], [13, 15, 2]]
data_diff = [2, 2, 2]
result = get_result(data_input, data_diff)
result = []

unfinish(data_input, result)
