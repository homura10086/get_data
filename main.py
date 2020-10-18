from get_ran import *
# interference problem
# mode0 正常
# mode1 杂散干扰
# mode2 邻道干扰
# mode3 交调干扰
# mode4 阻塞干扰
n = int(1e4)  # num_sample
with open("data.csv", "w", newline="") as datacsv:
	for i in range(n):
		mode = (i/n)*5
		Get_Data(i)
		# filename1 = "Config_" + str(modes[i])
		# Save_Config(filename1)
		# filename2 = "Perform_" + str(modes[i])
		# Save_Perform(filename2)
		Save_Data(datacsv, i)
		print(i+1, '/', n)
