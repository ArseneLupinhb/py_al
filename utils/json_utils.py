import json


def write_json(write_json, writepath):
	with open(writepath, "w", encoding='utf-8') as f:
		json.dump(write_json, f, ensure_ascii=False, sort_keys=False, indent=1, separators=(',', ': '))
	print("完成")


def read_json(read_path):
	with open(read_path, "r", encoding='utf-8') as f:
		data_json = json.load(f)
	# json.dump(write_json, f, ensure_ascii=False, sort_keys=False, indent=1, separators=(',', ': '))
	print("完成")
	return data_json
