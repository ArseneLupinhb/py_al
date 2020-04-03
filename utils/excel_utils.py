import xlrd
import xlwt


class excel_utils:
	@staticmethod
	def read_excel(read_path):
		path = r"C:\Users\AL\Desktop\test.xlsx"
		wb = xlrd.open_workbook(filename=path)
		print(wb.sheet_names())
		sheet1 = wb.sheet_by_index(0)
		print(sheet1.name, sheet1.nrows, sheet1.ncols)

	@staticmethod
	def write_excel(wirte_path):
		temp = xlwt.Workbook()
		sheet1 = temp.add_sheet('学生', cell_overwrite_ok=True)

	read_excel(r"C:\Users\AL\Desktop\test.xlsx")
