from get_ran import *
# coverage problem
# mode0 正常
# mode1 弱覆盖
# mode2 越区覆盖
# mode3 重叠覆盖
# mode4 覆盖不均衡

n = int(1e4)  # num_sample
with open("data.csv", "w", newline="") as datacsv:
	for i in range(n):
		mode = (i/n)*5  # for test
		Get_Data(i)
		# filename1 = "Config_" + str(modes[i])
		# Save_Config(filename1)
		# filename2 = "Perform_" + str(modes[i])
		# Save_Perform(filename2)
		Save_Data(datacsv, i)
		print(i+1, '/', n)
