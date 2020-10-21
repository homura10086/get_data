from get_ran import *
# capacity problem
# mode0 正常
# mode1 覆盖质量类(越区/重叠覆盖）
# mode2	切换类
# mode3	基础资源类

n = int(1e4)  # num_sample
with open("data.csv", "w", newline="") as datacsv:
	for i in range(n):
		# mode = (i/n)*4  # for test
		mode = randint(0, 3)
		Get_Data(i, int(mode))
		# filename1 = "Config_" + str(modes[i])
		# Save_Config(filename1)
		# filename2 = "Perform_" + str(modes[i])
		# Save_Perform(filename2)
		Save_Data(datacsv, i)
		print(i+1, '/', n)
