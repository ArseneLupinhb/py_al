import os
from datetime import datetime

import xlrd
import xlwt
from xlrd import xldate_as_tuple


class excel_tools:

	def write_diy(data_list, to_file_path):

		workbook = xlwt.Workbook()
		worksheet = workbook.add_sheet("sheet1", cell_overwrite_ok=True)

		all_row = 0
		write_row = 4
		for i in range(16):
			all_row = all_row + len(data_list[i])
			for j in range(len(data_list[i])):
				if (j >= 2):
					col_B = data_list[i][j][1]
					col_C = data_list[i][j][2]
					col_D = data_list[i][j][3]
					col_E = data_list[i][j][4]
					col_F = data_list[i][j][5]
					col_F = str(col_E) + str(col_F)
					col_G = data_list[i][j][6]
					col_H = data_list[i][j][7]
					print(str(col_B) + "--" + str(col_C) + '--' + str(col_D) + '--' + str(col_E) + '--' + str(
						col_F) + '--' + str(col_G) + '--' + str(col_H))
					worksheet.write(write_row, 1, col_D)
					worksheet.write(write_row, 2, "机械设备")
					worksheet.write(write_row, 4, col_C)
					worksheet.write(write_row, 5, col_E)
					worksheet.write(write_row, 6, col_H)
					worksheet.write(write_row, 7, col_F)
					worksheet.write(write_row, 14, col_B)
					worksheet.write(write_row, 16, col_G)
					worksheet.write(write_row, 18, "正常")
					worksheet.write(write_row, 19, "OK")
					write_row = write_row + 1

		workbook.save(to_file_path)
		print("done")

	def write_excel(data, outpath):
		'''
		just as the name of it write the data to the excel

		Args:
			data (list[tuple]): the data need to write into the excel
			outpath (): the path of the excel file generate


		Returns:
			1: suggest success(and only return the success ^_^)

		'''
		if (os.path.exists(outpath)):
			os.remove(outpath)

		workbook = xlwt.Workbook()
		worksheet = workbook.add_sheet("sheet1", cell_overwrite_ok=True)

		for tuple in data:
			for j in range(len(tuple)):
				print(str(data.index(tuple)) + '----' + str(j) + '------' + str(tuple[j]))
				worksheet.write(data.index(tuple), j, tuple[j])

		workbook.save(outpath)
		return 1

	def write_excel_DIY(data, outpath):
		"""
		just as the name of it write the data to the excel by DIY

		Parameters
		----------
			data (list[tuple]): the data need to write into the excel
			outpath (): the path of the excel file generate

		Returns
		-------
			1: suggest success(and only return the success ^_^)
		:Author:  AL
		:Create:  2019/11/12 18:48
		"""

		if (os.path.exists(outpath)):
			os.remove(outpath)

		workbook = xlwt.Workbook()
		worksheet = workbook.add_sheet("sheet1", cell_overwrite_ok=True)
		row_size = 0
		row = 0
		for tuple in data:
			for j in range(len(tuple)):

				if (j == 2):

					con3_str = str(tuple[j]).split(',')
					row_size = row_size + len(con3_str)
					for row_str in con3_str:
						worksheet.write(row, 0, tuple[0])
						worksheet.write(row, 1, tuple[1])
						worksheet.write(row, 2, row_str)
						worksheet.write(row, 3, str(tuple[3]))

						# it worked
						if (row > 65534):
							worksheet = workbook.add_sheet("sheet2", cell_overwrite_ok=True)
							row = 0
							workbook.save(outpath)

						row = row + 1

		workbook.save(outpath)

		return 1

	def write_excel_DIY_2(data, outpath):
		"""
		just as the name of it write the data to the excel by DIY

		Parameters
		----------
			data (list[tuple]): the data need to write into the excel
			outpath (): the path of the excel file generate

		Returns
		-------
			1: suggest success(and only return the success ^_^)
		:Author:  AL
		:Create:  2019/11/12 18:48
		"""

		if (os.path.exists(outpath)):
			os.remove(outpath)

		workbook = xlwt.Workbook()
		worksheet = workbook.add_sheet("sheet1", cell_overwrite_ok=True)

		row_size = 0
		row = 0

		for tuple in data:

			con3_str = str(tuple[3]).split(',')
			row_size = row_size + len(con3_str)

			for j in range(len(tuple)):

				if (j == 3):

					print(str(row_size) + '--------------------------------------------------------')
					for row_str in con3_str:
						worksheet.write(row, 3, row_str)
						# ok
						worksheet.write(row, 7, "OK")
						worksheet.write(row, 10, str(row + 1))

						print(str(tuple[0]) + '----' + str(tuple[1]) + '----' + str(
							tuple[2]) + '----' + row_str + '----' + str(
							tuple[4]) + '----' + str(tuple[5]))

						# it worked
						if (row > 65534):
							worksheet = workbook.add_sheet("sheet2", cell_overwrite_ok=True)
							row = 0
							workbook.save(outpath)

						row = row + 1
				# 设备编码
				if (j == 0):
					print(str(row_size - len(con3_str)) + '----' + str(row_size - 1) + '----' + str(tuple[0]))
					worksheet.write_merge(row_size - len(con3_str), row_size - 1, 0, 0, str(tuple[0]))

				# 设备名称
				if (j == 1):
					print(str(row_size - len(con3_str)) + '----' + str(row_size - 1) + '----' + str(tuple[1]))
					worksheet.write_merge(row_size - len(con3_str), row_size - 1, 1, 1, tuple[1])
					# 0 + 1
					worksheet.write_merge(row_size - len(con3_str), row_size - 1, 6, 6,
					                      str(tuple[1]) + '(' + str(tuple[0]) + ')')

				# 设备类型
				if (j == 2):
					print(str(row_size - len(con3_str)) + '----' + str(row_size - 1) + '----' + str(tuple[2]))
					if (tuple[2] == 1):
						# worksheet.write_merge(row_size - len(con3_str), row_size - 1, 2, 2, tuple[2])
						# tuple[2] = '测试设备'

						worksheet.write_merge(row_size - len(con3_str), row_size - 1, 2, 2, '测试设备')
					# worksheet.write_merge(row_size - len(con3_str), row_size - 1, 6, 6,
					# 					  '测试设备(' + str(tuple[0]) + ')')
					# worksheet.write_merge(row_size - len(con3_str), row_size - 1, , 6, '测试设备('+str(tuple[0])+')')

					if (tuple[2] == 5):
						# worksheet.write_merge(row_size - len(con3_str), row_size - 1, 2, 2, tuple[2])
						worksheet.write_merge(row_size - len(con3_str), row_size - 1, 2, 2, '机械设备')
				# worksheet.write_merge(row_size - len(con3_str), row_size - 1, 6, 6,
				# 					  '机械设备(' + str(tuple[0]) + ')')

				# 设备型号
				if (j == 4):
					print(str(row_size - len(con3_str)) + '----' + str(row_size - 1) + '----' + str(tuple[4]))
					worksheet.write_merge(row_size - len(con3_str), row_size - 1, 4, 4, tuple[4])

				# 时间
				if (j == 5):
					print(str(row_size - len(con3_str)) + '----' + str(row_size - 1) + '----' + str(tuple[5]))
					worksheet.write_merge(row_size - len(con3_str), row_size - 1, 5, 5, str(tuple[5]))
					# blank1
					worksheet.write_merge(row_size - len(con3_str), row_size - 1, 8, 8, '')
					# blank2
					worksheet.write_merge(row_size - len(con3_str), row_size - 1, 9, 9, '')

		workbook.save(outpath)

		return 1

	def read_excel(readpath):
		"""
			readpath(str): the read path of the excel file
		Returns:
			sheets_list(list): the list data of the sheets such as [[[]]]
		"""
		workbook = xlrd.open_workbook(readpath)
		sheets = workbook.sheets()

		sheets_list = []
		row_list = []

		for sheet in sheets:
			for i in range(sheet.nrows):
				row_content = []
				for j in range(sheet.ncols):
					ctype = sheet.cell(i, j).ctype  # 表格的数据类型
					cell = sheet.cell_value(i, j)
					# if ctype == 2 and cell % 1 == 0:  # 如果是整形
					# 	cell = int(cell)
					if ctype == 3:
						# 转成datetime对象
						date = datetime(*xldate_as_tuple(cell, 0))
						cell = date.strftime('%Y/%m/%d %H:%M:%S')
					# sheet.cell_value(i, j) = cell
					elif ctype == 4:
						cell = True if cell == 1 else False
					row_content.append(cell)
				# row_list.append(sheet.row_values(i))
				row_list.append(row_content)
			sheets_list.append(row_list)
			row_list = []
		return sheets_list

	def read_excel2(readpath):
		"""
			add the date change
			readpath(str): the read path of the excel file
		Returns:
			sheets_list(list): the list data of the sheets such as [[[]]]
		"""
		workbook = xlrd.open_workbook(readpath)
		sheets = workbook.sheets()

		sheets_list = []
		row_list = []

		for sheet in sheets:
			for i in range(sheet.nrows):
				row_content = []
				for j in range(sheet.ncols):
					ctype = sheet.cell(i, j).ctype  # 表格的数据类型
					cell = sheet.cell_value(i, j)
					# if ctype == 2 and cell % 1 == 0:  # 如果是整形
					# 	cell = int(cell)
					if ctype == 3:
						# 转成datetime对象
						date = datetime(*xldate_as_tuple(cell, 0))
						cell = date.strftime('%Y/%m/%d %H:%M:%S')
					# sheet.cell_value(i, j) = cell
					elif ctype == 4:
						cell = True if cell == 1 else False
					row_content.append(cell)
				# row_list.append(sheet.row_values(i))
				row_list.append(row_content)
			sheets_list.append(row_list)
			row_list = []
		return sheets_list
