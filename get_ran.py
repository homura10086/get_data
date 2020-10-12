import csv
from get_cpn import *
from copy import deepcopy
from random import randrange, choice
from numpy import random

lon, lan = 116.39, 39.9
ran = RanSubNetwork()
gnbs, dus, cus, nrs, celldus, rrus, antennas = [], [], [], [], [], [], []
cpe_te_sizes = []
CellState = ("Unknown", "Idle", "InActive", "Active")
OsType = ("Linux", "windows", "solaris")
VendorName = ("华为", "中兴", "诺基亚", "爱立信")
FreqBand1 = ("n1", "n2", "n3", "n5", "n7", "n8", "n12", "n14", "n18", "n20", "n25", "n26", "n28", "n29", "n30", "n34",
             "n38", "n39", "n40", "n41", "n48", "n50", "n51", "n53", "n65", "n66", "n70", "n71", "n74", "n75", "n76",
             "n77", "n78", "n79", "n80", "n81", "n82", "n83", "n84", "n86", "n89", "n90", "n91", "n92", "n93", "n94",
             "n95")
FreqBand2 = ("n257", "n258", "n260", "n261")
plmn = ("46000", "46002", "46004", "46007", "46001", "46006", "46009", "46003", "46005", "46011")
qci = (1, 2, 3, 4, 5, 6, 7, 8, 9, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 80, 82, 83, 84, 85, 86)
modes = []


def getplmn():
    i = randint(1, 3)
    if i == 1:
        return plmn[randint(0, 3)]
    elif i == 2:
        return plmn[randint(4, 6)]
    else:
        return plmn[randint(7, 9)]


s = getplmn()


def getPlmnList(s: str):
    i = plmn.index(s)
    if i <= 3:
        v = plmn[:4]
    elif i <= 6:
        v = plmn[4:7]
    else:
        v = plmn[7:]
    return v


def jw2xy(l: float, B: float):
    l = l * pi / 180
    B = B * pi / 180
    B0 = 30 * pi / 180
    a = 6378137
    b = 6356752.3142
    e = sqrt(1 - (b / a) * (b / a))
    e2 = sqrt((a / b) * (a / b) - 1)
    CosB0 = cos(B0)
    N = (a * a / b) / sqrt(1 + e2 * e2 * CosB0 * CosB0)
    K = N * CosB0
    SinB = sin(B)
    tans = tan(pi / 4 + B / 2)
    E2 = pow((1 - e * SinB) / (1 + e * SinB), e / 2)
    xx = tans * E2
    xc = K * log(xx)
    yc = K * l
    return xc, yc


def getSnaList(s: str, i: int):
    return [s + str(randint(0, 255)) + str(randint(0, 16777215)) for _ in range(i)]


def getRelatedCellDuList(i: int):
    return [x.CellDuId for x in dus[i].celldus]


# def getStrList(v, i):
#     l = []
#     while i != 0:
#         s = getValue(v)
#         flag = 0
#         for x in l:
#             if x == s:
#                 flag = 1
#                 i += 1
#                 break
#         if flag == 0:
#             l.append(s)
#         i -= 1
#     return l


def getTxPower(i: int, j: int, s: str):
    if s == "mean":
        return ran.gnbs[i].nrcells[j * 3].CellMeanTxPower + ran.gnbs[i].nrcells[j * 3 + 1].CellMeanTxPower + \
               ran.gnbs[i].nrcells[j * 3 + 2].CellMeanTxPower
    else:
        return ran.gnbs[i].nrcells[j * 3].CellMaxTxPower + ran.gnbs[i].nrcells[j * 3 + 1].CellMaxTxPower + \
               ran.gnbs[i].nrcells[j * 3 + 2].CellMaxTxPower


def shannon(bw: int, power: int):
    s = 10 * log10(power * normal(0.5, 0.03) * 1E3)
    return bw * 1E6 * log2(1 + (s + 174 - 10 * log10(bw * 1E6)))


def get_nrcell_configuration(i: int, nrcellsize: int, mode_temp: int):
    bw1 = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
    # bw2 = (50, 100, 200, 400)
    # bw_sul = (5, 10, 15, 20, 25, 30, 40)
    dl1 = [randrange(422000, 434000, 20), randrange(386000, 398000, 20), randrange(361000, 376000, 20),
           randrange(173800, 178800, 20), randrange(524000, 538000, 20), randrange(185000, 192000, 20),
           randrange(145800, 149200, 20), randrange(151600, 153600, 20), randrange(172000, 175000, 20),
           randrange(158200, 164200, 20), randrange(386000, 399000, 20), randrange(171800, 178800, 20),
           randrange(151600, 160600, 20), randrange(470000, 472000, 20), randrange(402000, 405000, 20),
           randrange(514000, 524000, 20), randrange(376000, 384000, 20), randrange(460000, 480000, 20),
           randrange(499200, 537999, 3), randrange(499200, 537999, 6), randrange(636667, 646666, 1),
           randrange(636668, 646666, 2), randrange(286400, 303400, 20), randrange(285400, 286400, 20),
           randrange(496700, 499000, 20), randrange(422000, 440000, 20), randrange(399000, 404000, 20),
           randrange(123400, 130400, 20), randrange(295000, 303600, 20), randrange(620000, 680000, 1),
           randrange(620000, 680000, 2), randrange(620000, 653333, 1), randrange(620000, 653332, 2),
           randrange(693334, 733333, 1), randrange(693334, 733332, 2), randrange(499200, 538000, 20)]
    # dl2{ getStep(2054166,2104165,1),getStep(2054167,2104165,2),getStep(2016667,2070832,1),
    # getStep(2016667,2070831,2),getStep(2229166,2279165,1), getStep(2229167,2279165,2),
    # getStep(2070833,2084999,1),getStep(2070833,2084999,2) },
    ul1 = [randrange(384000, 396000, 20), randrange(370000, 382000, 20), randrange(342000, 357000, 20),
           randrange(164800, 169800, 20), randrange(500000, 514000, 20), randrange(176000, 178300, 20),
           randrange(139800, 143200, 20), randrange(157600, 159600, 20), randrange(163000, 166000, 20),
           randrange(166400, 172400, 20), randrange(370000, 383000, 20), randrange(162800, 169800, 20),
           randrange(140600, 149600, 20), randrange(461000, 463000, 20), randrange(402000, 405000, 20),
           randrange(514000, 524000, 20), randrange(376000, 384000, 20), randrange(460000, 480000, 20),
           randrange(499200, 537999, 3), randrange(499200, 537999, 6), randrange(636667, 646666, 1),
           randrange(636668, 646666, 2), randrange(286400, 303400, 20), randrange(285400, 286400, 20),
           randrange(496700, 499000, 20), randrange(384000, 402000, 20), randrange(342000, 356000, 20),
           randrange(339000, 342000, 20), randrange(132600, 139600, 20), randrange(285400, 294000, 20),
           randrange(620000, 680000, 1), randrange(620000, 680000, 2), randrange(620000, 653333, 1),
           randrange(620000, 653332, 2), randrange(693334, 733333, 1), randrange(693334, 733332, 2),
           randrange(499200, 537996, 6), randrange(499200, 538000, 20), randrange(176000, 183000, 20)]
    # ul2 = dl2,
    # sul{ getStep(342000,357000),getStep(176000,183000),getStep(166400,172400),getStep(140600,149600),
    # getStep(384000,396000),getStep(342000,356000),getStep(164800,169800),getStep(402000,405000) }
    dl1.sort()
    ul1.sort()
    '''Here ! ! !'''
    rand_indexs = sample(range(6), randint(0, 6))
    '''here'''
    for k in range(nrcellsize):
        n = NrCell()
        n.NrCellId = "GNB" + getId(i) + "/NrCell" + getId(k)
        # n.NCGI = n.NrCellId
        # n.CellState = getValue(CellState)
        n.S_NSSAIList = getSnaList(s, randint(1, 8))
        # n.NrTAC = to_string(getRandom(0, 65535))
        '''Here! ! !'''
        if k in rand_indexs and (i != 0 or mode_temp != 2):  # 每个Ran的第一个gnb无mode2(切换）问题
            mode = mode_temp
        else:
            mode = 0
        '''here'''
        modes.append(mode)
        n.ArfcnDL = choice(dl1)
        n.ArfcnUL = choice(ul1)
        # //n.ArfcnSUL = getValue(sul)
        if mode == 3:  # 基础资源类
            n.BsChannelBwDL = choice(bw1[:len(bw1) // 2])
            n.BsChannelBwUL = choice(bw1[:len(bw1) // 2])
        else:
            n.BsChannelBwDL = choice(bw1)
            n.BsChannelBwUL = choice(bw1)
        # //n.BsChannelBwSUL = getValue(bw_sul)
        # else {
        #     n.ArfcnDL = getValue(dl2)
        #     n.ArfcnUL = getValue(ul2)
        #     n.BsChannelBwDL = getBsChannelBw(bw2)
        #     n.BsChannelBwUL = getBsChannelBw(bw2)
        # }
        # //n.relatedBwp = "Bwp-" + to_string(getRandom(0, 3));
        nrs.append(n)


def get_nrcell_performance():
    k = 0
    for i, x in enumerate(ran.gnbs):
        for j, y in enumerate(x.nrcells):
            te_size = cpe_te_sizes[k] + len(cpns[k].terminals) + len(cpns[k].cpes)
            y.ConnMax = te_size
            y.ConnMean = round(y.ConnMax / normal(2, 0.05))
            mode = modes[k]
            if mode == 2:  # 切换类
                rand_index = randint(0, 5)
                if modes[(i - 1) * 6 + rand_index] == 2:  # 所选小区本来就是切换类问题小区
                    pass
                else:
                    modes[(i - 1) * 6 + rand_index] = 2
                    ran.gnbs[i - 1].nrcells[rand_index].AttOutExecInterXn = \
                        round(normal(21, 1) * ran.gnbs[i - 1].nrcells[rand_index].ConnMean)
                y.AttOutExecInterXn = round(normal(21, 1) * y.ConnMean)
            else:
                y.AttOutExecInterXn = round(normal(15, 1) * y.ConnMean)
            y.UpOctDL = shannon(gnbs[i].nrcells[j].BsChannelBwDL,
                                rrus[k // 3].antennas[k % 3].MaxTxPower) / (8 * 1024) * normal(0.5, 0.03) * y.ConnMax
            y.SuccOutInterXn = round(y.AttOutExecInterXn * uniform(0.9, 1))
            y.UpOctUL = y.UpOctDL / normal(8, 1)
            y.ULMeanNL = round(normal(-110, 3))
            y.ULMaxNL = round(y.ULMeanNL / normal(1.2, 0.1))
            y.NbrPktDL = round(y.UpOctDL / normal(1, 0.1))
            y.NbrPktUL = round(y.UpOctUL / normal(1, 0.1))
            y.NbrPktLossDL = round(uniform(0, 0.1) * y.NbrPktDL)
            y.CellMaxTxPower = rrus[k // 3].antennas[k % 3].MaxTxPower / normal(20, 3)
            y.CellMeanTxPower = y.CellMaxTxPower * normal(0.5, 0.03)
            k += 1
    cpe_te_sizes.clear()


def get_gnb(i: int, gnbsize: int, mode: int):
    nrcellsize = 6
    for k in range(gnbsize):
        g = GNBFunction()
        g.GNBId = "Ran" + getId(i) + "/GNB" + getId(k)
        g.GNBName = "GNB-" + ''.join(sample(ascii_letters + digits, 3))
        # gnbs.GNBGId = s + gnbs.GNBId
        g.Longitude, g.Latitude = getjw(jws, lon, lan, 0.005, 0.006, 0.7)
        # gnbs.bwp = get_bwp(j)
        get_nrcell_configuration(k, nrcellsize, mode)
        g.nrcells = deepcopy(nrs)
        nrs.clear()
        gnbs.append(g)


def get_celldu(i: int, j: int, celldusize: int):
    for k in range(celldusize):
        c = CellDu()
        c.CellDuId = "GNB" + getId(i) + "/DU" + getId(j) + "/CellDu" + getId(k)
        # celldus.NCGI = s + "GNB" + getId(i) + "/" + "NrCell" + getId(j)
        # celldus.CellState = getValue(CellState);
        c.S_NSSAIList = gnbs[i].nrcells[j * 3 + k].S_NSSAIList
        if 1:
            c.ArfcnDL = gnbs[i].nrcells[j * 3 + k].ArfcnDL
            c.ArfcnUL = gnbs[i].nrcells[j * 3 + k].ArfcnUL
            # celldus.ArfcnSUL = getValue(sul)
            c.BsChannelBwDL = gnbs[i].nrcells[j * 3 + k].BsChannelBwDL
            c.BsChannelBwUL = gnbs[i].nrcells[j * 3 + k].BsChannelBwUL
            # celldus.BsChannelBwSUL = getValue(bw_sul)
        # else {
        #     c.ArfcnDL = getValue(dl2)
        #     c.ArfcnUL = getValue(ul2)
        #     c.BsChannelBwDL = getValue(bw1)
        #     c.BsChannelBwUL = getValue(bw1)
        # }
        # celldus.relatedBwp = "Bwp-" + to_string(getRandom(0, 3))
        celldus.append(c)


def get_du(dusize: int):
    celldusize = 3
    for j in range(len(gnbs)):
        for k in range(dusize):
            d = DuFunction()
            d.DuId = "GNB" + getId(j) + "/DU" + getId(k)
            d.DuName = "DU-" + ''.join(sample(ascii_letters + digits, 3))
            d.Longitude, d.Latitude = getjw(jws, gnbs[j].Longitude, gnbs[j].Latitude, 0.004, 0.005, 0.5)
            # d.bwp = get_bwp(1)
            get_celldu(j, k, celldusize)
            d.celldus = deepcopy(celldus)
            celldus.clear()
            dus.append(d)


def get_cu(i: int, cusize: int):
    for k in range(cusize):
        c = CuFunction()
        c.CuId = "Ran" + getId(i) + "/CU" + getId(k)
        c.CuName = "CU-" + ''.join(sample(ascii_letters + digits, 3))
        # getPlmnList(cus.PLMNIDList, s)
        c.Longitude = gnbs[k].Longitude
        c.Latitude = gnbs[k].Latitude
        # cus.cucp = get_cucp(i, j, s)
        # cus.cuup = get_cuup(i, getRandom(1, 3))
        cus.append(c)


def get_antenna(i: int, j: int, f: bool, antennasize: int):
    # SupportSeq = ("410MHz-7125MHz", "24250MHz-52600MHz")
    for k in range(antennasize):
        a = Antenna()
        a.AntennaId = "GNB" + getId(i) + "/Rru" + getId(j) + "/Antenna" + getId(k)
        a.AntennaName = "Antenna-" + ''.join(sample(ascii_letters + digits, 3))
        mode = modes[i * 6 + j * 3 + k]
        if mode == 1:  # 覆盖质量类
            a.MaxTxPower = round(normal(330, 15))
            a.MaxTiltValue = randint(100, 1800)
        else:
            a.MaxTxPower = round(normal(240, 15))
            a.MaxTiltValue = randint(1800, 3600)
        a.MinTiltValue = randint(1, a.MaxTiltValue - 1)
        a.RetTilt = randint(a.MinTiltValue, a.MaxTiltValue)
        # if (f) {
        #     antennas.SupportedSeq = SupportSeq[0]
        #     antennas.ChannelInfo = getStrList(FreqBand1, getRandom(1, 47))
        # }
        # else {
        #     antennas.SupportedSeq = SupportSeq[1]
        #     antennas.ChannelInfo = getStrList(FreqBand2, getRandom(1, 4))
        # }
        # int j = getRandom(1, 64)
        # antennas.beam = get_beam(i, j)
        cpe_te_size = get_cpn(i, j, k, dus[j].Longitude, dus[j].Latitude, a.MaxTxPower,
                              gnbs[i].nrcells[j * 3 + k].BsChannelBwDL, mode)
        cpe_te_sizes.append(cpe_te_size)
        antennas.append(a)


def get_rru_configuration(rrusize: int):
    antennasize = 3
    for j in range(len(gnbs)):
        for k in range(rrusize):
            r = Rru()
            r.RruId = "GNB" + getId(j) + "/Rru" + getId(k)
            r.RruName = "Rru-" + ''.join(sample(ascii_letters + digits, 3))
            # rrus.VendorName = getValue(VendorName)
            # rrus.SerialNumber = "Rru-" + rrus.VendorName + "-" + getId(i)
            # rrus.VersionNumber = to_string(getRandom(1, 3)) + "." + getrandoms(1)
            # rrus.DateOfLastService = getTime()
            f = bool(randint(0, 1))
            # if (f) { rrus.FreqBand1 = getStrList(FreqBand1, getRandom(1, 47)); }
            # else { rrus.FreqBand1 = getStrList(FreqBand2, getRandom(1, 4)); }
            r.relatedCellDuList = getRelatedCellDuList(j * 2 + k)
            get_antenna(j, k, f, antennasize)
            r.antennas = deepcopy(antennas)
            antennas.clear()
            rrus.append(r)


def get_rru_performance(rrusize: int):
    for j in range(len(gnbs)):
        for k in range(rrusize):
            rrus[j * rrusize + k].MeanTxPower = getTxPower(j, k, "mean")
            rrus[j * rrusize + k].MaxTxPower = getTxPower(j, k, "max")
            rrus[j * rrusize + k].MeanPower = rrus[j * rrusize + k].MeanTxPower * normal(10, 0.1)


def Get_Data(i: int, mode: int = 0):
    dusize = 2
    gnbsize = 3
    rrusize = dusize
    cusize = gnbsize
    ran.RanSNId = "Ran" + getId(i)
    ran.RanSNName = "Ran-" + ''.join(sample(ascii_letters + digits, 3))
    get_gnb(i, gnbsize, mode)
    ran.gnbs = deepcopy(gnbs)
    get_du(dusize)
    ran.dus = deepcopy(dus)
    jws.clear()
    get_cu(i, cusize)
    ran.cus = deepcopy(cus)
    get_rru_configuration(rrusize)
    get_nrcell_performance()
    get_rru_performance(rrusize)
    ran.rrus = deepcopy(rrus)
    gnbs.clear()
    dus.clear()
    cus.clear()
    rrus.clear()


def save_nrcell(v: list, csvwriter):
    for x in v:
        csvwriter.writerow([x.NrCellId, x.S_NSSAIList, x.ArfcnDL, x.ArfcnUL, x.BsChannelBwDL, x.BsChannelBwUL])


def save_cedu(v: list, csvwriter):
    for x in v:
        csvwriter.writerow([x.CellDuId, x.S_NSSAIList, x.ArfcnDL, x.ArfcnUL, x.BsChannelBwDL, x.BsChannelBwUL])


def save_du(v: list, csvwriter):
    csvwriter.writerow(["DuId", "DuName", "Longitude", "Latitude"])
    for x in v:
        csvwriter.writerow([x.DuId, x.DuName, x.Longitude, x.Latitude])
    csvwriter.writerow(["CellDuId", "SNSSAIList", "ArfcnDL", "ArfcnUL", "BsChannelBwDL", "BsChannelBwUL"])
    for x in v:
        save_cedu(x.celldus, csvwriter)
    save_cpn(csvwriter)
    # p << "BwpContext" << "," << "IsInitalBwp" << "," << "SubCarrierSpacing" << "," <<
    #     "CyclicPrefix" << "," << "StartRB" << "," << "NumOfRBs" << endl;
    # for (auto x : v) {
    #     save_bwp(x.bwp, p);
    # }


def save_antenna(v: list, csvwriter):
    for x in v:
        csvwriter.writerow([x.AntennaId, x.AntennaName, x.RetTilt, x.MaxTiltValue, x.MinTiltValue, x.MaxTxPower])


def save_rru(v: list, csvwriter):
    csvwriter.writerow(["RruId", "RruName", "relatedCellDuList"])
    for x in v:
        csvwriter.writerow([x.RruId, x.RruName, x.relatedCellDuList])
    csvwriter.writerow(["AntennaId", "AntennaName", "RetTilt", "MaxTilt", "MinTiltValue", "MaxTxPower"])
    for x in v:
        save_antenna(x.antennas, csvwriter)


def save_gnb(v: list, csvwriter):
    csvwriter.writerow(["GNBId", "GNBName", "Longitude", "Latitude"])
    for x in v:
        csvwriter.writerow([x.GNBId, x.GNBName, x.Longitude, x.Latitude])
    csvwriter.writerow(["NrCellId", "S-NSSAIList", "ArfcnDL", "ArfcnUL", "BsChannelBwDL", "BsChannelBwUL"])
    for x in v:
        save_nrcell(x.nrcells, csvwriter)


def save_cu(v: list, csvwriter):
    csvwriter.writerow(["CuId", "CuName", "Longitude", "Latitude"])
    for x in v:
        csvwriter.writerow([x.CuId, x.CuName, x.Longitude, x.Latitude])
        # savelist(x.cus.PLMNIDList, p)


def Save_Config(filename: str):
    with open(filename + ".csv", "a", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect="excel")
        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        csvwriter.writerow(["RanSNId", "RanSNName"])
        csvwriter.writerow([ran.RanSNId, ran.RanSNName])
        save_gnb(ran.gnbs, csvwriter)
        # p << "BwpContext" << "," << "IsInitalBwp" << "," << "SubCarrierSpacing" << "," <<
        #     "CyclicPrefix" << "," << "StartRB" << "," << "NumOfRBs"
        # save_bwp(x.gnbs.bwp, p)
        save_du(ran.dus, csvwriter)
        save_cu(ran.cus, csvwriter)
        # p << "CuCPId" << "," << "DiscardTimer" << endl
        # p << x.cus.cucp.CuCpId << ","
        # savelist(x.cus.cucp.DiscardTimer, p)
        # save_cuup(x.cus.cuup, p)
        # save_cellcu(x.cus.cucp.cellcu, p)
        save_rru(ran.rrus, csvwriter)
        # save_beam(x.rru.antenna.beam, p)
        # datacsv.close()


def save_nrcell_performance(csvwriter):
    csvwriter.writerow(["ULMeanNL", "ULMaxNL", "UpOctUL", "UpOctDL", "NbrPktUL", "NbrPktDL", "NbrPktLossDL",
                        "CellMeanTxPower", "CellMaxTxPower", "ConnMean", "ConnMax", "AttOutExecInterXn",
                        "SuccOutInterXn"])
    for x in ran.gnbs:
        for y in x.nrcells:
            csvwriter.writerow([y.ULMeanNL, y.ULMaxNL, y.UpOctUL, y.UpOctDL, y.NbrPktUL, y.NbrPktDL, y.NbrPktLossDL,
                                y.CellMeanTxPower, y.CellMaxTxPower, y.ConnMean, y.ConnMax, y.AttOutExecInterXn,
                                y.SuccOutInterXn])


def save_rru_performance(csvwriter):
    csvwriter.writerow(["MeanTxPower", "MaxTxPower", "MeanPower"])
    for x in ran.rrus:
        csvwriter.writerow([x.MeanTxPower, x.MaxTxPower, x.MeanPower])


def Save_Perform(filename: str):
    with open(filename + ".csv", "a", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect="excel")
        save_cpn_performance(csvwriter)
        save_nrcell_performance(csvwriter)
        save_rru_performance(csvwriter)
        # datacsv.close()


# def save_node(ofstream& p):
#     int i = 1;
#     float x1, y1;
#     for (auto x : ran.cu) {
#         jw2xy(x.Longitude, x.Latitude, x1, y1);
#         p << i++ << ',' << x1 << ',' << y1 << endl;
#     }
#     for (auto x : ran.du) {
#         jw2xy(x.Longitude, x.Latitude, x1, y1);
#         p << i++ << ',' << x1 << ',' << y1 << endl;
#     }
#     save_node_cpn(p, i);


def Save_Data(datacsv, f: int):
    writer = csv.writer(datacsv, dialect="excel")
    if f == 0:
        writer.writerow(
            ["BsChannelBwUL", "BsChannelBwDL", "ConnMean", "AttOutExecInterXn", "UpOctUL", "MaxTxPower",
             "RetTilt", "RSRP", "RSRQ", "TransRatePeak", "Label"])
    temps = Temp()
    for gnb in ran.gnbs:
        for nrcell in gnb.nrcells:
            temps.nrs.append(nrcell)
    for rru in ran.rrus:
        for antenna in rru.antennas:
            temps.ans.append(antenna)
    for i in range(len(temps.nrs)):
        writer.writerow(
            [temps.nrs[i].BsChannelBwUL, temps.nrs[i].BsChannelBwDL, temps.nrs[i].ConnMean,
             temps.nrs[i].AttOutExecInterXn, temps.nrs[i].UpOctUL, temps.ans[i].MaxTxPower, temps.ans[i].RetTilt,
             round(cpns[i].RSRP_mean), round(cpns[i].RSRQ_mean), round(cpns[i].TransRate_mean), modes[i]])
        if i % 6 == 5:
            writer.writerow('')
    modes.clear()
    writer.writerow('')
