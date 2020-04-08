def get_max_diff(data_input):
	max_diff = 0
	for i in range(len(data_input)):
		if (data_input[i][2] > max_diff):
			max_diff = data_input[i][2]
	return max_diff


def get_result(data_input):
	result = []
	for j in range(len(data_input)):
		max_diff = 0
		if (len(data_input) > 0):
			max_diff = get_max_diff(data_input)
		else:
			break
		for i in range(len(data_input)):
			if (max_diff == data_input[i][2]):
				if (len(result) == 0):
					result.append(data_input[i])
					data_input.remove(data_input[i])
					break
				else:
					flag = 1
					for k in range(len(result)):
						if (data_input[i][1] < result[k][0] or data_input[i][0] > result[k][1]):
							flag = 1
						else:
							flag = 0
							data_input.remove(data_input[i])
							break
					if (flag == 1):
						result.append(data_input[i])
						data_input.remove(data_input[i])
						break
					else:
						break
	return result


def get_input():
	a = []
	while True:
		try:
			input_str = input()
			s = int(input_str.split(",")[0])
			f = int(input_str.split(",")[1])
			if (s == 0 and f == 0):
				break
			else:
				a.append([s, f, f - s])
		except:
			break
	return a


def out_result(result_list):
	for i in result_list:
		print(str(i[0]) + "," + str(i[1]))


data_input = get_input()
result_list = get_result(data_input)
out_result(result_list)
