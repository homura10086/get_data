import ran_interference
import ran_capacity
import ran_coverage
import multiprocessing as mp
import time
from random import randint, choices

# coverage problem
'''
    label 0 正常
    label 1 弱覆盖
    label 2 越区覆盖
    label 3 重叠覆盖
    label 4 覆盖不均衡
'''
# capacity problem
'''
    label 0   正常
    label 5   覆盖质量类(越区/重叠覆盖）
    label 6   切换类
    label 7   基础资源类
'''
# interference problem
'''
    label 0   正常
    label 8   杂散干扰
    label 9   邻道干扰
    label 10   阻塞干扰
'''

num_core = 3
num_sample_all = int(3e4)


def main(batch: int, n: int):
    with open("data" + str(batch) + ".csv", "w", newline="") as datacsv:
        if batch == 1:
            for i in range(0, n):  # coverage problem
                # label = int((i / n) * 5)  # for test
                label = choices([i for i in range(5)], weights=(1, 3, 3, 3, 3), k=1)[0]
                ran_coverage.Get_Data(i, label)
                ran_coverage.Save_Data(datacsv, bool(i))
                print((i + 1) * num_core, '/', num_sample_all)

        # label = int((i / n) * 4)  # for test
        elif batch == 2:
            for i in range(0, n):  # capacity problem
                label = choices([i for i in range(4)], weights=(1, 3, 3, 3), k=1)[0]
                ran_capacity.Get_Data(i, label)
                ran_capacity.Save_Data(datacsv, bool(i))
        else:
            for i in range(0, n):  # interference problem
                label = choices([i for i in range(4)], weights=(1, 3, 3, 3), k=1)[-1]
                ran_interference.Get_Data(i, label)
                ran_interference.Save_Data(datacsv, bool(i))
                print((i + 1) * num_core, '/', num_sample_all)
        # # filename1 = "Config_" + str(modes[i])
        # # Save_Config(filename1)
        # # filename2 = "Perform_" + str(modes[i])
        # # Save_Perform(filename2)


if __name__ == '__main__':
    start = time.time()
    # p1 = mp.Process(target=main, args=(1, num_sample_all // num_core))
    # p2 = mp.Process(target=main, args=(2, num_sample_all // num_core))
    p3 = mp.Process(target=main, args=(3, num_sample_all // num_core))
    # p4 = mp.Process(target=main, args=(4, num_sample // num_core))
    # p1.start()
    # p2.start()
    p3.start()
    # p4.start()
    # p1.join()
    # p2.join()
    p3.join()
    # p4.join()
    print(round(time.time() - start))
