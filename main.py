from get_ran import *
# interference problem

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
