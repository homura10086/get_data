class Jwd:
    def __init__(self):
        self.lon, self.lat = 0.0, 0.0


class Temp:
    def __init__(self):
        self.nrs, self.ans = [], []


class Terminal:
    def __init__(self):
        self.TerminalId, self.TerminalType, self.TerminalBrand = '', '', ''
        self.Storage, self.Computing,  = 0, 0
        self.Longitude, self.Latitude = 0.0, 0.0


class Cpe:
    def __init__(self):
        self.CpeId, self.CpeName, self.SupportedType = '', '', ''
        self.MaxDistance, self.TransRateMean, self.TransRatePeak, self.RSRP, self.RSRQ = 0, 0, 0, 0, 0
        self.Longitude, self.Latitude = 0.0, 0.0
        self.terminals = []


class CpnSubNetwork:
    def __init__(self):
        self.CpnSNId, self.CpnSNName = '', ''
        self.cpes, self.terminals = [], []
        self.RSRP_mean, self.RSRQ_mean, self.TransRate_mean = 0, 0, 0


class NrCell:
    def __init__(self):
        self.NrCellId, self.NCGI, self.CellState, self.NrTAC, self.relatedBwp = '', '', 'Unknow', '', ''
        self.S_NSSAIList = []  # Mcc+Mnc+SST+SD
        self.ArfcnDL, self.ArfcnUL, self.ArfcnSUL, self.BsChannelBwDL, self.BsChannelBwUL, self.BsChannelBwSUL, \
            self.ULMeanNL, self.ULMaxNL, self.NbrPktUL, self.NbrPktDL, self.NbrPktLossDL, self.ConnMean, self.ConnMax, \
            self.AttOutExecInterXn, self.SuccOutInterXn = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.UpOctUL, self.UpOctDL, self.CellMeanTxPower, self.CellMaxTxPower = 0.0, 0.0, 0.0, 0.0


class CuFunction:
    def __init__(self):
        self.CuId, self.CuName = '', ''
        self.PLMNIDList = []  # 标识符列表Mcc+Mnc
        self.Longitude, self.Latitude = 0.0, 0.0
        self.cucps, self.cuups = [], []


class CellDu:
    def __init__(self):
        self.CellDuId, NCGI, CellState, relatedBwp = '', '', 'Unknow', ''
        self.S_NSSAIList = []
        self.ArfcnDL, self.ArfcnUL, self.ArfcnSUL, self.BsChannelBwDL, self.BsChannelBwUL, \
            self.BsChannelBwSUL = 0, 0, 0, 0, 0, 0


class DuFunction:
    def __init__(self):
        self.DuId, self.DuName = '', ''
        self.Longitude, self.Latitude = 0.0, 0.0
        self.bwps, self.celldus = [], []


class GNBFunction:
    def __init__(self):
        self.GNBId, self.GNBName, self.GNBGId = '', '', ''
        self.Longitude, self.Latitude = 0.0, 0.0
        self.nrcells, self.bwps = [], []


class Antenna:
    def __init__(self):
        self.AntennaId, self.AntennaName, self.SupportedSeq = '', '', ''
        self.RetTilt, self.MaxTiltValue, self.MinTiltValue, self.MaxTxPower = 0, 0, 0, 0
        self.ChannelInfo = []
        self.beams = []


class Rru:
    def __init__(self):
        self.RruId, self.RruName, self.VendorName, self.SerialNumber, self.VersionNumber, \
            self.DateOfLastService = '', '', '', '', '', ''
        self.relatedCellDuList, self.FreqBand = [], []
        self.antennas = []
        self.MeanTxPower, self.MaxTxPower, self.MeanPower = 0.0, 0.0, 0.0


class RanSubNetwork:
    def __init__(self):
        self.RanSNId, self.RanSNName = '', ''
        self.gnbs, self.dus, self.cus, self.rrus = [], [], [], []
