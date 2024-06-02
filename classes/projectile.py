import numpy as np


class Projectile:
    def __init__(
            self,
            name,
            m_g=0,
            m_grs=0,
            cal_mm=0,
            cal_inch=0,
            Cd=0.5,
            SD_m=0,
            SD_inch=0
            ):
        self.name = name
        if (m_g != 0):
            self.m_g = m_g
            self.m_grs = m_g*15.4323584
        elif (m_grs != 0):
            self.m_grs = m_grs
            self.m_g = m_grs/15.4323584
        else:
            raise Exception("No mass specified")

        if (cal_mm != 0):
            self.cal_mm = cal_mm
            self.cal_inch = cal_mm*0.0393700787
        elif (cal_inch != 0):
            self.cal_inch = cal_inch
            self.cal_mm = cal_inch/0.0393700787
        else:
            raise Exception("No caliber specified")

        self.Cd = Cd
        if (SD_m != 0):
            self.SD_m = SD_m
            self.SD_inch = SD_m/703.06958
        elif (SD_inch != 0):
            self.SD_inch = SD_inch
            self.SD_m = SD_inch*703.06958
        else:
            self.SD_inch = (self.m_grs*0.000142857143) / \
                (np.pi*(self.cal_inch/2)**2)
            self.SD_m = (self.m_g/1000)/(np.pi*(self.cal_mm/2000)**2)

        self.A = np.pi*(self.cal_mm/2000)**2
        self.m_kg = self.m_g/1000

    def __str__(self):
        ret = f"""{self.name}
        Caliber: {str(self.cal_inch)[1:]} ({self.cal_mm:.2f})
        Mass (g): {self.m_g:.2f} g
        Mass (grs): {self.m_grs} grs

        A: {self.A},
        SD (m/kg): {self.SD_m}, SD (in/lbs): {self.SD_inch}"""
        return ret

    def __repr__(self):
        return self.name
