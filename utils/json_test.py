import datetime
import json


class DateEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, datetime.datetime):
			return obj.strftime('%Y-%m-%d %H:%M:%S')

		elif isinstance(obj, datetime.date):
			return obj.strftime("%Y-%m-%d")

		else:
			return json.JSONEncoder.default(self, obj)


def write_json(write_json, writepath):
	with open(writepath, "w", encoding='utf-8') as f:
		# write_json = json.dumps(write_json, cls=DateEncoder)
		# print(write_json)
		json.dump(write_json, f, ensure_ascii=False, sort_keys=False, indent=4, separators=(',', ': '), cls=DateEncoder)
	print("完成")
