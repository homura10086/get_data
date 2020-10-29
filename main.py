from get_ran import *
import multiprocessing as mp
import time

# interference problem
# mode0 正常
# mode1 杂散干扰
# mode2 邻道干扰
# mode3 阻塞干扰
num_core = 4
num_sample = int(1e4 // num_core)  # num_sample


def main(batch: int, n: int):
    with open("interference_data" + str(batch) + ".csv", "w", newline="") as datacsv:
        for i in range(n):
            # mode = int((i/n)*4)  # for test
            Get_Data(i)
            # filename1 = "Config_" + str(modes[i])
            # Save_Config(filename1)
            # filename2 = "Perform_" + str(modes[i])
            # Save_Perform(filename2)
            Save_Data(datacsv, bool(i))
            if batch == 1:
                print((i + 1) * num_core, '/', num_sample * num_core)


if __name__ == '__main__':
    start = time.time()
    p1 = mp.Process(target=main, args=(1, num_sample))
    p2 = mp.Process(target=main, args=(2, num_sample))
    p3 = mp.Process(target=main, args=(3, num_sample))
    p4 = mp.Process(target=main, args=(4, num_sample))
    # p5 = mp.Process(target=main, args=(5, num_sample))
    # p6 = mp.Process(target=main, args=(6, num_sample))
    # p7 = mp.Process(target=main, args=(7, num_sample))
    # p8 = mp.Process(target=main, args=(8, num_sample))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    # p5.start()
    # p6.start()
    # p7.start()
    # p8.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    # p5.join()
    # p6.join()
    # p7.join()
    # p8.join()
    # main(1, num_sample)
    print(round(time.time() - start, 1))
