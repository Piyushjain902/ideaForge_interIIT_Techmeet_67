
# Drone Control and IMU Data Recording with MAVSDK

This project demonstrates how to control a drone's roll, pitch, yaw, and throttle using MAVSDK, and simultaneously record Inertial Measurement Unit (IMU) data (accelerometer, gyroscope, and magnetometer readings). The script enables real-time drone control and telemetry data logging for analysis.

---

## Features
- Sends roll, pitch, yaw, and throttle commands to the drone using MAVSDK.
- Records real-time IMU data, including accelerometer, gyroscope, and magnetometer readings.
- Provides a robust interface for analyzing drone behavior during controlled flights.

---

## Prerequisites
- **Python 3.8+**
- A working installation of **MAVSDK** for Python.
- A drone simulator (e.g., PX4 SITL) or hardware drone connected via MAVSDK.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Required Python Packages**:
   Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   Example dependencies:
   ```text
   asyncio
   mavsdk
   numpy
   ```

3. **Set Up the Drone Simulator**:
   If using PX4 SITL, ensure it is running and accessible via the MAVSDK UDP connection.

---

## Usage

1. **Run the Script**:
   Execute the script in your Python environment:
   ```bash
   python script_name.py
   ```

2. **Expected Functionality**:
   - The script connects to the drone and begins sending roll, pitch, yaw, and throttle commands.
   - IMU data, including accelerometer, gyroscope, and magnetometer readings, is recorded in real time.

3. **Adjusting Commands**:
   Modify the roll, pitch, yaw, and throttle values in the script to test various drone movements and record their effects.

---

## Script Overview

1. **Drone Connection**:
   The script connects to the drone using MAVSDK:
   ```python
   drone = System()
   await drone.connect(system_address="udp://:14540")
   ```

2. **Sending Roll, Pitch, Yaw Commands**:
   Commands for roll, pitch, yaw, and throttle are sent using MAVSDK's manual control capabilities:
   ```python
   await drone.manual_control.set_manual_control_input(pitch, roll, yaw, throttle)
   ```

3. **IMU Data Recording**:
   The script records accelerometer, gyroscope, and magnetometer data using the telemetry API:
   ```python
   async for imu in drone.telemetry.imu():
       accel_data = [imu.acceleration_frd.forward_m_s2, ...]
       gyro_data = [imu.angular_velocity_frd.forward_rad_s, ...]
       mag_data = [imu.magnetic_field_frd.forward_gauss, ...]
   ```

---

## Requirements.txt Example
Here’s an example of what your `requirements.txt` should include:
```text
asyncio
mavsdk
numpy
```

---

## Notes
- Ensure that the drone is in a safe environment before testing manual control commands.
- Use appropriate safety measures to prevent damage or accidents during flight testing.

---

## License
Include any licensing information here.

---
