from math import *
from random import randint, uniform, sample
from string import ascii_letters, digits
from numpy.random import normal, choice
from data import *
from copy import deepcopy

EARTH_RADIUS = 6378.137
cpns = []
cpes = []
terminals = []
jws = []


def getId(i: int):
    return "0" * (3 - len(str(i + 1))) + str(i + 1)


# def getStorage(s):
#     storage = (1,16,32,64,128,256,512,1024,1152,1280,1536,2048,2560)
#     if s == "手机":
#         return storage[round(normal(4, 0.5))]
#     elif s == "Pad":
#         return storage[round(normal(4.5, 1))]
#     elif s == "电脑":
#         return storage[round(normal(9, 1))]
#     else:
#         return storage[0]


def getSupportedType():
    st = ("4G", "5G", "6G")
    return choice(st, p=(0.6, 0.3, 0.1))


def getDistance(lng1: float, lat1: float, lng2: float, lat2: float):
    radLat1 = lat1 * pi / 180.0
    radLat2 = lat2 * pi / 180.0
    a = radLat1 - radLat2
    b = lng1 * pi / 180.0 - lng2 * pi / 180.0
    dst = 2 * asin((sqrt(pow(sin(a / 2), 2) + cos(radLat1) * cos(radLat2) * pow(sin(b / 2), 2))))
    dst = dst * EARTH_RADIUS
    dst = round(dst * 10000) / 10000
    return dst


def getjw(v: list, x: float, y: float, dx1: float, dx2: float, dis1: float):
    i = randint(0, 3)
    if i == 0:
        x1 = x + uniform(dx1, dx2)
        y1 = y + uniform(dx1, dx2)
    elif i == 1:
        x1 = x - uniform(dx1, dx2)
        y1 = y + uniform(dx1, dx2)
    elif i == 2:
        x1 = x - uniform(dx1, dx2)
        y1 = y - uniform(dx1, dx2)
    else:
        x1 = x + uniform(dx1, dx2)
        y1 = y - uniform(dx1, dx2)
    if dis1 > 0:
        if len(v) > 0:
            for e in v:
                if getDistance(x1, y1, e.lon, e.lat) < dis1:
                    x1, y1 = getjw(v, x1, y1, dx1, dx2, dis1)
        temp = Jwd()
        temp.lon = x
        temp.lat = y
        v.append(temp)
    return x1, y1


def getTransRate(s: str):
    if s == "4G":
        return round(normal(150, 10))
    elif s == "5G":
        return round(normal(500, 20))
    else:
        return round(normal(1000, 100))


def Mean(res: int, val: int, count: int, cpe_size: int):
    if count == (cpe_size-1):
        return (res+val)/cpe_size
    else:
        return res+val


def get_terminal(i: int, j: int, k: int, l: int, terminalsize: int, x: float, y: float):
    for e in range(terminalsize):
        t = Terminal()
        t.Longitude, t.Latitude = getjw(jws, x, y, 0.001, 0.002, 0)
        if l >= 0:  # CPE接入
            t.TerminalId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k) + "/Cpe" + getId(l) + "/Terminal" + \
                           getId(e)
        else:  # 直接接入
            t.TerminalId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k) + "/Terminal" + getId(e)
        # t.TerminalType = getTerminalType(s)
        # t.TerminalBrand = getTerminalBrand(t.TerminalType)
        # t.Storage = getStorage(t.TerminalType)
        # t.Computing = getComputing(t.TerminalType)
        terminals.append(t)


def get_cpe(i: int, j: int, k: int, x: float, y: float, MaxTxPower: int, mode: int):
    cpe_size = randint(3, 5)
    TransRatePeak_mean, RSRP_mean, RSRQ_mean = 0, 0, 0
    for e in range(cpe_size):
        cpe = Cpe()
        cpe.CpeId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k) + "/Cpe" + getId(e)
        cpe.CpeName = "Cpe-" + ''.join(sample(ascii_letters + digits, 3))
        cpe.MaxDistance = round(normal(100, 10))
        cpe.SupportedType = getSupportedType()
        cpe.Longitude, cpe.Latitude = getjw(jws, x, y, 0.001, 0.003, 0.1)
        lamuda = (getDistance(x, y, cpe.Longitude, cpe.Latitude) / 0.28) * (240.0 / MaxTxPower)
        cpe.TransRatePeak = round(getTransRate(cpe.SupportedType) * (1 + 0.2 * (1 - lamuda)))
        cpe.TransRateMean = round(cpe.TransRatePeak * normal(0.5, 0.03))
        if mode == 0 or mode == 4:
            cpe.RSRP = round(normal(-95, 1.5) * (1 + 0.1 * (lamuda - 1)))
        elif mode == 1:
            cpe.RSRP = round(normal(-105, 1.5) * (1 + 0.1 * (lamuda - 1)))
        else:
            cpe.RSRP = round(normal(-85, 1.5) * (1 + 0.1 * (lamuda - 1)))
        cpe.RSRQ = round(-11.25 * (1 + 0.4 * (lamuda - 1)))
        TransRatePeak_mean = Mean(TransRatePeak_mean, cpe.TransRatePeak, e, cpe_size)
        RSRP_mean = Mean(RSRP_mean, cpe.RSRP, e, cpe_size)
        RSRQ_mean = Mean(RSRQ_mean, cpe.RSRQ, e, cpe_size)
        get_terminal(i, j, k, e, randint(2, 3), cpe.Longitude, cpe.Latitude)
        cpe.terminals = deepcopy(terminals)
        terminals.clear()
        cpes.append(cpe)
    return TransRatePeak_mean, RSRP_mean, RSRQ_mean


def get_cpn(i: int, j: int, k: int, x: float, y: float, cpe_te_size: int, MaxTxPower: int, mode: int):
    c = CpnSubNetwork()
    c.CpnSNId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k)
    c.CpnSNName = "Cpn-" + ''.join(sample(ascii_letters + digits, 3))
    c.TransRate_mean, c.RSRP_mean, c.RSRQ_mean = get_cpe(i, j, k, x, y, MaxTxPower, mode)
    c.cpes = deepcopy(cpes)
    cpes.clear()
    for t in c.cpes:
        cpe_te_size += len(t.terminals)
    if mode == 1:  # 弱覆盖
        get_terminal(i, j, k, -1, randint(1, 5), x, y)
    elif mode == 2 or mode == 3:  # 越区覆盖、重叠覆盖
        get_terminal(i, j, k, -1, randint(10, 15), x, y)
    else:
        get_terminal(i, j, k, -1, randint(5, 10), x, y)
    c.terminals = deepcopy(terminals)
    terminals.clear()
    if len(cpns) == 18:
        cpns.clear()
        jws.clear()
    cpns.append(c)
    return cpe_te_size


def save_terminal(csvwriter, t: list):
    for x in t:
        csvwriter.writerow([x.TerminalId, x.Longitude, x.Latitude])


def save_cpn(csvwriter):
    csvwriter.writerow(["CpnSNId", "CpnSNName"])
    for x in cpns:
        csvwriter.writerow([x.CpnSNId, x.CpnSNName])
    csvwriter.writerow(["CpeId", "CpeName", "MaxDistance", "SupportedType", "Longitude", "Latitude"])
    for x in cpns:
        for y in x.cpes:
            csvwriter.writerow([y.CpeId, y.CpeName, y.MaxDistance, y.SupportedType, y.Longitude, y.Latitude])
    csvwriter.writerow(["TerminalId", "Longitude", "Latitude"])
    for x in cpns:
        for y in x.cpes:
            save_terminal(csvwriter, y.terminals)
        save_terminal(csvwriter, x.terminals)


def save_cpn_performance(csvwriter):
    csvwriter.writerow(["TransRateMean", "TransRatePeak", "RSRP", "RSRQ"])
    for x in cpns:
        for y in x.cpes:
            csvwriter.writerow([y.TransRateMean, y.TransRatePeak, y.RSRP, y.RSRQ])

# def save_node_cpn(ofstream& p, int& i) {
#     float x1, y1;
#     for (auto x : cpns) {
#         for (auto y : x.terminal) {
#             jw2xy(y.Longitude, y.Latitude, x1, y1);
#             p << i++ << ',' << x1 << ',' << y1 << endl;
#         }
#         for (auto y : x.cpe) {
#             jw2xy(y.Longitude, y.Latitude, x1, y1);
#             p << i++ << ',' << x1 << ',' << y1 << endl;
#             for (auto z : y.terminal) {
#                 jw2xy(z.Longitude, z.Latitude, x1, y1);
#                 p << i++ << ',' << x1 << ',' << y1 << endl;
#             }
#         }
#     }
# }
# def save_edge_cpn(ofstream& p, int& dunum, const int& terminalnum) {
#     int i = terminalnum, count = 0;
#     for (auto x : cpns) {
#         count++;
#         for (auto y : x.terminal) {
#             p << dunum << ',' << i++ << endl;
#         }
#         for (auto y : x.cpe) {
#             int cpenum = i;
#             p << dunum << ',' << i++ << endl;
#             for (auto z : y.terminal) {
#                 p << cpenum << ',' << i++ << endl;
#             }
#         }
#         if (count % 3 == 0) { dunum++; }
#     }
# }

