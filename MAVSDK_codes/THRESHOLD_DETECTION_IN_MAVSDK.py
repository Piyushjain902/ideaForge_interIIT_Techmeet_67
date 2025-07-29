import asyncio
from mavsdk import System
from mavsdk.failure import Failure,FailureType,FailureUnit
import time


    #FLAGS FOR MOTOR FAILED
async def inject_failure(drone):
    # Wait for 10 seconds before injecting the failure
    await asyncio.sleep(10)
    print("-- Injecting Motor failure,failing motor 1 by external command")
    await drone.failure.inject(
        FailureUnit.SYSTEM_MOTOR, FailureType.OFF, instance=1
    )

async def get_imu_data():
    start_time = 0
    end_time = 0
    # Connect to the drone
    drone = System()
    await drone.connect(system_address="udp://:14540")

    # Wait for the drone to connect
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone is connected!")
            break

    telemetry = drone.telemetry

    # Set the rate at which IMU data is updated (in Hz)
    await telemetry.set_rate_imu(200.0)

    # Flags for detecting failure conditions for each motor
    roll_failure_motor1 = False
    pitch_failure_motor1 = False
    yaw_failure_motor1 = False
    
    roll_failure_motor2 = False
    pitch_failure_motor2 = False
    yaw_failure_motor2 = False

    roll_failure_motor3 = False
    pitch_failure_motor3 = False
    yaw_failure_motor4 = False

    roll_failure_motor4 = False
    pitch_failure_motor4 = False
    yaw_failure_motor4 = False

    altitude_failure = False
    external_failure = False
    accelero_fail=False


    M1_F = False
    M2_F = False
    M3_F = False
    M4_F = False

    asyncio.create_task(inject_failure(drone))
    start_time = time.time()




    print("Fetching IMU data...")
    async for imu in telemetry.imu():
        # Process sensor data (gyro and accelerometer readings)
        gyro_data = [imu.angular_velocity_frd.forward_rad_s,imu.angular_velocity_frd.right_rad_s,imu.angular_velocity_frd.down_rad_s] # Gyroscope data
        accel_data = [imu.acceleration_frd.forward_m_s2,imu.acceleration_frd.right_m_s2,imu.acceleration_frd.down_m_s2]  # Accelerometer data

        if M1_F or M2_F or M3_F or M4_F:
            end_time = time.time()
            break

        if accel_data[2]>-7.5:
            accelero_fail = True

#-------------------------------------------------#
# STARTING WITH MOTOR 1                          
#-------------------------------------------------#

# Motor 1(CASE HOVER)

        if gyro_data[0] > 1.2:
            roll_failure_motor1 = True
        else:
            roll_failure_motor1 = False

        if gyro_data[1]<-0.1:
            pitch_failure_motor1=True
        else:
            pitch_failure_motor1=False

        if gyro_data[2] <-3.0:
            yaw_failure_motor1 = True
        else:
            yaw_failure_motor1 = False

# # Motor 1(CASE ROLL)

#         if gyro_data[0] <-0.5:
#             roll_failure_motor1 = True
#         else:
#             roll_failure_motor1 = False

#         if gyro_data[1]<0:
#             pitch_failure_motor1=True
#         else:
#             pitch_failure_motor1=False

#         if gyro_data[2] < -1.0:
#             yaw_failure_motor1 = True
#         else:
#             yaw_failure_motor1 = False


# # Motor 1(CASE PITCH)

        if gyro_data[0] >1.2:   # isme ek bar dekh lena wo <-0.1 bhi chalega
            roll_failure_motor1 = True
        else:
            roll_failure_motor1 = False

        if gyro_data[1]<-1.2:
            pitch_failure_motor1=True
        else:
            pitch_failure_motor1=False

        if gyro_data[2] < -1.2:
            yaw_failure_motor1 = True
        else:
            yaw_failure_motor1 = False

# # Motor 1(CASE YAW)

#         if gyro_data[0] >1.0:
#             roll_failure_motor1 = True
#         else:
#             roll_failure_motor1 = False

#         if gyro_data[1]>0:
#             pitch_failure_motor1=True
#         else:
#             pitch_failure_motor1=False

#         if gyro_data[2] < -1.8:
#             yaw_failure_motor1 = True
#         else:
#             yaw_failure_motor1 = False


#-------- Log failure if both roll and pitch conditions are met for motor 1
        if roll_failure_motor1 and pitch_failure_motor1 and yaw_failure_motor1 and accelero_fail:
            print("MOTOR-1 FAILED")
            M1_F = True

        if M1_F or M2_F or M3_F or M4_F:
            end_time = time.time()
            break


#-------------------------------------------------#
# STARTING WITH MOTOR 2                            
#------------------------------------------------#


# Motor 2(CASE HOVER)

        if  gyro_data[0]>-0.15:
            roll_failure_motor2 = True
        else:
            roll_failure_motor2 = False

        if gyro_data[1]>0.1:
            pitch_failure_motor2=True
        else:
            pitch_failure_motor2=False

        if gyro_data[2] < -3.0:
            yaw_failure_motor2 = True
        else:
            yaw_failure_motor2 = False

# # Motor 2(CASE ROLL)

        if gyro_data[0] < -2.5:
            roll_failure_motor2 = True
        else:
            roll_failure_motor2 = False

        if gyro_data[1]>2.5:
            pitch_failure_motor2=True
        else:
            pitch_failure_motor2=False

        if gyro_data[2] < -1.7:
            yaw_failure_motor2 = True
        else:
            yaw_failure_motor2 = False

# # Motor 2(CASE PITCH)

#         if gyro_data[0] <-5.0:
#             roll_failure_motor2 = True
#         else:
#             roll_failure_motor2 = False

#         if gyro_data[1]>1.0:
#             pitch_failure_motor2=True
#         else:
#             pitch_failure_motor2=False

#         if gyro_data[2] < -2:
#             yaw_failure_motor2 = True
#         else:
#             yaw_failure_motor2 = False

# # Motor 2(CASE YAW)

#         if gyro_data[0] > 0:
#             roll_failure_motor2 = True
#         else:
#             roll_failure_motor2 = False

#         if gyro_data[1]>0:
#             pitch_failure_motor2=True
#         else:
#             pitch_failure_motor2=False

#         if gyro_data[2] < -1.5:
#             yaw_failure_motor2 = True
#         else:
#             yaw_failure_motor2 = False


#-------# Log failure for motor 2 if all conditions are met
        if roll_failure_motor2 and pitch_failure_motor2 and yaw_failure_motor2 and accelero_fail:
            print("MOTOR-2 FAILED")
            M2_F = True

        if M1_F or M2_F or M3_F or M4_F:
            end_time = time.time()
            break


#-------------------------------------------------#
# STARTING WITH MOTOR 3                           
#-------------------------------------------------#


# Motor 3(CASE HOVER)

        if gyro_data[0] <-1.0:
            roll_failure_motor3 = True
        else:
            roll_failure_motor3 = False

        if gyro_data[1]<-0.1:
            pitch_failure_motor3=True
        else:
            pitch_failure_motor3=False

        if gyro_data[2] >2.0:
            yaw_failure_motor3 = True
        else:
            yaw_failure_motor3 = False

# # Motor 3(CASE ROLL)

#         if gyro_data[0] <-1:
#             roll_failure_motor3 = True
#         else:
#             roll_failure_motor3 = False

#         if gyro_data[1]<-0.5:
#             pitch_failure_motor3=True
#         else:
#             pitch_failure_motor3=False

#         if gyro_data[2] >1:
#             yaw_failure_motor3 = True
#         else:
#             yaw_failure_motor3 = False


# # Motor 3(CASE PITCH)

        if gyro_data[0] <-0.7:
            roll_failure_motor3 = True
        else:
            roll_failure_motor3 = False

        if gyro_data[1]<-2:
            pitch_failure_motor3=True
        else:
            pitch_failure_motor3=False

        if gyro_data[2] >1.2:
            yaw_failure_motor3 = True
        else:
            yaw_failure_motor3 = False

# # Motor 3(CASE YAW)  # yeh last mat badla hai

        if gyro_data[0] <-0.5:
            roll_failure_motor3 = True
        else:
            roll_failure_motor3 = False

        if gyro_data[1]<0:
            pitch_failure_motor3=True
        else:
            pitch_failure_motor3=False

        if gyro_data[2] >-3:
            yaw_failure_motor3 = True
        else:
            yaw_failure_motor3 = False

 

#------ Log failure for motor 3 if all conditions are met
        if roll_failure_motor3 and pitch_failure_motor3 and yaw_failure_motor3 and accelero_fail:
            print("MOTOR-3 FAILED")
            M3_F= True

        if M1_F or M2_F or M3_F or M4_F:
            end_time = time.time()
            break


#-------------------------------------------------#
# STARTING WITH MOTOR 4                            
#-------------------------------------------------#

# Motor 4(CASE HOVER)

        if gyro_data[0] > 0.13:
            roll_failure_motor4 = True
        else:
            roll_failure_motor4 = False

        if gyro_data[1]>0.1:
            pitch_failure_motor4=True
        else:
            pitch_failure_motor4=False

        if gyro_data[2] >0.5:
            yaw_failure_motor4 = True
        else:
            yaw_failure_motor4 = False


# # Motor 4(CASE ROLL)

#         if gyro_data[0] > 1.0:
#             roll_failure_motor4 = True
#         else:
#             roll_failure_motor4 = False

#         if gyro_data[1]>1.0:
#             pitch_failure_motor4=True
#         else:
#             pitch_failure_motor4=False

#         if gyro_data[2] < -1.0:
#             yaw_failure_motor4 = True
#         else:
#             yaw_failure_motor4 = False


# # Motor 4(CASE PITCH)

#         if gyro_data[0] > 2.0:
#             roll_failure_motor4 = True
#         else:
#             roll_failure_motor4 = False

#         if gyro_data[1]<-0.80:
#             pitch_failure_motor4=True
#         else:
#             pitch_failure_motor4=False

#         if gyro_data[2] >1:
#             yaw_failure_motor4 = True
#         else:
#             yaw_failure_motor4 = False


# # Motor 4(CASE YAW)

#         if gyro_data[0] > 0.1:
#             roll_failure_motor4 = True
#         else:
#             roll_failure_motor4 = False

#         if gyro_data[1]>1:
#             pitch_failure_motor4=True
#         else:
#             pitch_failure_motor4=False

#         if gyro_data[2] >3:
#             yaw_failure_motor4 = True
#         else:
#             yaw_failure_motor4 = False


 #------- Log failure for motor 4 if all conditions are met
        if roll_failure_motor4 and pitch_failure_motor4 and yaw_failure_motor4 and accelero_fail:
            print("MOTOR-4 FAILED")
            M4_F = True

        if M1_F or M2_F or M3_F or M4_F:
            end_time = time.time()
            break


    # now we can move on control when we know the FAILED MOTOR

    if(M1_F):
        pass
    if(M2_F):
        pass
    if(M3_F):
        pass
    if(M4_F):
        pass 
    print("THE MEASURED LATENCY IS")
    print(end_time-start_time-10)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_imu_data())
