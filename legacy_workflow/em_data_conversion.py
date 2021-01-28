'''
Created on 14/6/2020
@author: Neil Symington

This script is for converting aseg-gdf EM data to a netcdf file. The netcdf file will also include some additional
AEM system metadata.
'''

from geophys_utils.netcdf_converter import aseg_gdf2netcdf_converter
import netCDF4
import os, math
import numpy as np
# SO we can see the logging. This enables us to debug
import logging
import sys
sys.path.append('../scripts')
from aem_utils import AEM_System
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")


def rollpitchyaw_array(roll, pitch, yaw):
    """
    FRom Ross' rollpitchyaw_matrix.m
    """

    d2r = math.pi / 180
    # Calculate the cosine and sine of role, pitch and yaw
    cosr = np.cos(d2r * roll);
    cosp = np.cos(d2r * pitch);
    cosy = np.cos(d2r * yaw);

    sinr = np.sin(d2r * roll);
    sinp = np.sin(d2r * pitch);
    siny = np.sin(d2r * yaw)
    # Create the array
    a = np.zeros(shape=(roll.shape[0], 3, 3), dtype=np.float32)
    a[:, 0, 0] = cosp * cosy
    a[:, 0, 1] = cosp * siny
    a[:, 0, 2] = -sinp
    a[:, 1, 0] = sinr * sinp * cosy - cosr * siny
    a[:, 1, 1] = sinr * sinp * siny + cosr * cosy
    a[:, 1, 2] = sinr * cosp
    a[:, 2, 0] = cosr * sinp * cosy + sinr * siny
    a[:, 2, 1] = cosr * sinp * siny - sinr * cosy
    a[:, 2, 2] = cosr * cosp

    return a


# Define paths

root = '/home/nsymington/Documents/GA/AEM/EM"

nc_out_path = os.path.join(root, "AUS_10024_InJune_EM_MGA55.nc")
#nc_out_path = os.path.join(root, "AUS_10024_Orana_MGA56.nc")
#nc_out_path = os.path.join(root, "AUS_10023_SSC_EM_MGA53.nc")

dat_in_path = os.path.join(root, 'ASEG_gdf', 'AUS_10024_InJune_EM_MGA55.dat')
#dat_in_path = os.path.join(root, 'ASEG_gdf', 'AUS_10024_Orana_EM_MGA56.dat')
#dat_in_path = r"C:\Users\symin\OneDrive\Documents\GA\AEM\delivered\Delivered_20171113\01_EM\AUS_10023_SouthernStuart_EM\AUS_10023_SouthernStuart_EM.dat"

dfn_in_path = os.path.join(root, 'ASEG_gdf', 'AUS_10024_InJune_EM_MGA55.dfn')
#dfn_in_path = os.path.join(root, 'ASEG_gdf', 'AUS_10024_Orana_EM_MGA56.dfn')
#dfn_in_path = r"C:\Users\symin\OneDrive\Documents\GA\AEM\delivered\Delivered_20171113\01_EM\AUS_10023_SouthernStuart_EM\AUS_10023_SouthernStuart_EM.dfn"

# GDA94 MGA zone 55
crs_string = "EPSG:28355"

# Open the lm and hm files
root = r"C:\Users\symin\OneDrive\Documents\GA\AEM\STM"

lm_file = os.path.join(root, "Skytem312Fast-LM_pV.stm")

hm_file = os.path.join(root, "Skytem312Fast-HM_pV.stm")

# Initialise instance of ASEG2GDF netcdf converter

d2n = aseg_gdf2netcdf_converter.ASEGGDF2NetCDFConverter(nc_out_path,
                                                 dat_in_path,
                                                 dfn_in_path,
                                                 crs_string,
                                                 fix_precision=True,
                                                 remove_null_columns = True)
d2n.convert2netcdf()

# Now open the file

d = netCDF4.Dataset(nc_out_path, "a")

# Create an AEM system instance
skytem = AEM_System("SkyTEM312Fast", dual_moment = True)


# Parse
skytem.parse_stm_file(lm_file, 'LM')

skytem.parse_stm_file(hm_file, 'HM')

lm_a = skytem.LM['Transmitter']['WaveFormCurrent']
hm_a = skytem.HM['Transmitter']['WaveFormCurrent']


# Dimensions for the current times
lm_current_times = d.createDimension("low_moment_current_time",
                                    skytem.LM['Transmitter']['WaveFormCurrent'].shape[0])

hm_current_times = d.createDimension("high_moment_current_time",
                                    skytem.HM['Transmitter']['WaveFormCurrent'].shape[0])

# Create dimension variables

lm_current_times = d.createVariable("low_moment_current_time","f8",("low_moment_current_time",))
hm_current_times = d.createVariable("high_moment_current_time","f8",("high_moment_current_time",))

lm_waverform_current = d.createVariable("low_moment_waverform_current","f8",("low_moment_current_time",))

hm_waverform_current = d.createVariable("high_moment_waverform_current","f8",("high_moment_current_time",))

lm_current = skytem.LM['Transmitter']['WaveFormCurrent']
hm_current = skytem.HM['Transmitter']['WaveFormCurrent']

lm_current_times[:] = lm_current[:,0]
hm_current_times[:] = hm_current[:,0]

lm_waverform_current[:] = lm_current[:,1]
hm_waverform_current[:] = hm_current[:,1]

# Add an scalar vats for frame geometries assuming a horizontal frame
# These are from the SkyTEM files

tx_area = d.createVariable('tx_loop_area',"f8",())
tx_area[:] = 337.
tx_area.units = 'm**2'
tx_area.long_name = "Transmitter (Tx) Loop Area"

gate_openclose = d.createDimension("gate_open_close",
                                   2)
gates = d.createVariable("gate_open_close","i1",("gate_open_close",))

lm_window_times =  d.createVariable("low_moment_window_time","f8",("low_moment_gate",
                                                                  "gate_open_close"))

hm_window_times =  d.createVariable("high_moment_window_time","f8",("high_moment_gate",
                                                                   "gate_open_close"))

# Add data

lm_gates = skytem.LM['Receiver']['WindowTimes']
hm_gates = skytem.HM['Receiver']['WindowTimes']

lm_window_times[:] = lm_gates
hm_window_times[:] = hm_gates

rx_x_pos = d.createVariable("Rx_z_component_position_x","f8",())
rx_x_pos[:] = -13.37
rx_x_pos.units = 'm'
rx_x_pos.long_name = 'Z-component EM sensor relative position from centre of horizontal frame, in flight direction'
rx_x_pos.sign_convention = 'Front of frame is positive'

rx_y_pos = d.createVariable("Rx_z_component_position_y","f8",())
rx_y_pos[:] = 0.
rx_y_pos.units = 'm'
rx_y_pos.long_name = 'Z-component EM sensor relative position from centre of horizontal frame, perpendicular to flight direction'
rx_y_pos.sign_convention = 'Port of frame is positive' # GA-AEM conventions

rx_z_pos = d.createVariable("Rx_z_component_position_z","f8",())
rx_z_pos[:] = 2. # GA-AEM conventions
rx_z_pos.units = 'm'
rx_z_pos.long_name = 'Z-component EM sensor relative position from centre of horizontal frame, in vertical direction'
rx_z_pos.sign_convention = 'Up is positive'

rx_x_x_pos = d.createVariable("Rx_x_component_position_x","f8",())
rx_x_x_pos[:] = -14.75
rx_x_x_pos.units = 'm'
rx_x_x_pos.long_name = 'X-component EM sensor relative position from centre of horizontal frame, in flight direction'
rx_x_x_pos.sign_convention = 'Front of frame is positive'

rx_x_y_pos = d.createVariable("Rx_x_component_position_y","f8",())
rx_x_y_pos[:] = 0.
rx_x_y_pos.units = 'm'
rx_x_y_pos.long_name = 'X-component EM sensor relative position from centre of horizontal frame, perpendicular to flight direction'
rx_x_y_pos.sign_convention = 'Starboard of frame is positive'

rx_x_z_pos = d.createVariable("Rx_x_component_position_z","f8",())
rx_x_z_pos[:] = -0.04
rx_x_z_pos.units = 'm'
rx_x_z_pos.long_name = 'X-component EM sensor relative position from centre of horizontal frame, in vertical direction'
rx_x_z_pos.sign_convention = 'Down is positive'



lm_waverform_current.units = 'normalised_current_amplitude'
hm_waverform_current.units = 'normalised_current_amplitude'

lm_current_times.units = 'seconds'
hm_current_times.units = 'seconds'

lm_window_times.units = 'seconds_since_waveform_rampdown'
hm_window_times.units = 'seconds_since_waveform_rampdown'

# Now we want to calculate the roll pitch and yaw

#Input angle X and Y in the SkyTEM sign convention
#Nose up      = +ve x-tilt
#Left wing up = +ve y-tilt

#output roll and pitch are in the GA modelling sign convention
#Left wing up = +ve roll
#Nose down is = +ve pitch
#turn left is = +ve yaw

# As frame roll is simply y-tilt we just rename the variable
try:
    d.renameVariable('y_tilt', 'roll')
except KeyError:
    print('Variable not found')

# We need to recalculate for the GA convention
d2r = math.pi/180
r2d = 180/math.pi
try:
    # From Ross' tilt2rollpitchyaw function
    frame_pitch = -r2d * np.arcsin(np.sin(d2r*d.variables['x_tilt'][:])/
                                   np.cos(d2r*d.variables['roll'][:]))
    d.variables['x_tilt'][:] = frame_pitch
    d.renameVariable('x_tilt', 'pitch')
except KeyError:
    print('Variable not found')

# Yaw is zero so we can just create a scalar variable
yaw =  d.createVariable("yaw","f8",())
yaw[:] = 0

# Add an attribute describing the sign convention
pitch = d['pitch']
pitch.sign_convention = 'Nose down is positive pitch'

roll = d['roll']
pitch.sign_convention = 'Left wing up is positive roll'

d['yaw'].long_name = 'Rotation of frame vertical axis'

# Get the offsets for the receiver

yaw_arr = yaw[:]*np.ones(shape=roll[:].shape,
              dtype = roll[:].dtype)

R = rollpitchyaw_array(roll[:], pitch[:],
                       yaw_arr)
# Get the offsets of the receiver when the frame is horizontal
txrx_dx = d['Rx_z_component_position_x'][:]
txrx_dy = d['Rx_z_component_position_y'][:]
txrx_dz = d['Rx_z_component_position_z'][:]

v0 = np.array([txrx_dx, txrx_dy, txrx_dz])

# Get matrix products of two arrays

V = np.matmul(v0, R)

# Create variables

txrx_dx = d.createVariable("TxRx_dx","f8",('point'))
txrx_dx[:] = V[:,0]
txrx_dx.units = 'm'
txrx_dx.aseg_gdf_format = 'E7.2'
txrx_dx.long_name = 'Z-component EM sensor relative position from centre of frame, in flight direction'
txrx_dx.sign_convention = 'Front of frame is positive'

txrx_dy = d.createVariable("TxRx_dy","f8",('point'))
txrx_dy[:] = V[:,1]
txrx_dy.units = 'm'
txrx_dy.aseg_gdf_format = 'E7.2'
txrx_dy.long_name = 'X-component EM sensor relative position from centre of frame, perpendicular to flight direction'
txrx_dy.sign_convention = 'Starboard of frame is positive'

txrx_dz = d.createVariable("TxRx_dz","f8",('point'))
txrx_dz[:] = V[:,2]
txrx_dz.units = 'm'
txrx_dz.aseg_gdf_format = 'E7.2'
txrx_dz.long_name = 'Z-component EM sensor relative position from centre of frame, in vertical direction'
txrx_dz.sign_convention = 'Down is positive'

d.close()
