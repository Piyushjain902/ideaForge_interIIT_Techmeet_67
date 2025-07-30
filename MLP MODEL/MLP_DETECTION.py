import asyncio
from mavsdk import System
import os
import joblib
import numpy as np


# Load the MLP model from the .joblib file
def load_mlp_model(model_name,scalar_name):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, model_name)
    file_path_2 = os.path.join(script_dir, scalar_name)
    mlp_model = joblib.load(file_path)
    scalar=joblib.load(file_path_2)
    return mlp_model,scalar

mlp_model,scalar = load_mlp_model('MLP_MODEL_FOR_NOISE.joblib','SCALER_FOR_MLP_MODEL_WITH_NOISE.joblib')


async def get_imu_data():
    # Connect to the drone
    drone = System()
    await drone.connect(system_address="udp://:14540")
    await asyncio.sleep(4)

    # Wait for the drone to connect
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone is connected!")
            break

    telemetry = drone.telemetry

    # Set the rate at which IMU data is updated (in Hz)
    await telemetry.set_rate_imu(200.0)

    # Fetch and print IMU data
    print("Fetching IMU data...")
    async for imu in telemetry.imu():
        gyro_data = [imu.angular_velocity_frd.forward_rad_s,imu.angular_velocity_frd.right_rad_s,imu.angular_velocity_frd.down_rad_s] # Gyroscope data
        accel_data = [imu.acceleration_frd.forward_m_s2,imu.acceleration_frd.right_m_s2,imu.acceleration_frd.down_m_s2]  # Accelerometer data
        mag_data = [imu.magnetic_field_frd.forward_gauss,imu.magnetic_field_frd.right_gauss,imu.magnetic_field_frd.down_gauss]# Magnetometer data

        input_data = np.array([[accel_data[0],accel_data[1],accel_data[2],gyro_data[0],gyro_data[1],gyro_data[2],mag_data[0],mag_data[1],mag_data[2]]])
        scaled_data=scalar.transform(input_data)
        predicted_value = mlp_model.predict(scaled_data)
        if(predicted_value!= 0):
            print(predicted_value)
            break

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_imu_data())
