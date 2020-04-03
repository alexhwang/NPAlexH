# -*- coding: utf-8 -*-
"""
Created on Fri Dec 06 12:06:16 2019

@author: Hera
"""
if __name__ == '__main__':
    import os
    import visa
    from nplab.instrument.spectrometer.seabreeze import OceanOpticsSpectrometer
    from nplab import datafile

    os.chdir(r'C:\Users\aykh2\Documents')    
    
    rm= visa.ResourceManager()  
    spec = OceanOpticsSpectrometer(0) 

    # wutter = Uniblitz("COM8")
    spec.show_gui(blocking = False)
    df = datafile.current()    
    df.show_gui(blocking = False)
