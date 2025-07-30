
# Drone IMU Data Processing with MLP Model

This project processes IMU (Inertial Measurement Unit) data from a drone in real time, using a pre-trained **MLP (Multi-Layer Perceptron)** model to predict specific values based on sensor readings. The script connects to a drone via MAVSDK and utilizes gyroscope, accelerometer, and magnetometer data to make predictions.

## Features
- Connects to a drone using MAVSDK via a UDP connection.
- Fetches real-time IMU data (Gyroscope, Accelerometer, and Magnetometer).
- Uses a pre-trained MLP model and scaler to process and predict outcomes based on sensor data.
- Stops execution when a specific condition is met (`predicted_value != 0`).

---

## Prerequisites
- **Python 3.8+**
- A working installation of **MAVSDK** for Python.
- Pre-trained **MLP model** and **scaler** saved as `.joblib` files.
- A drone simulator or real drone accessible via MAVSDK (e.g., PX4 SITL or hardware drone).

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
   joblib
   numpy
   ```

3. **Set Up the Drone Simulator**:
   If using PX4 SITL, ensure it is running and accessible via the MAVSDK UDP connection.

4. **Add the Model and Scaler Files**:
   - Place the `MLP_MODEL_FOR_NOISE.joblib` and `SCALER_FOR_MLP_MODEL_WITH_NOISE.joblib` files in the project directory.

---

## Usage

1. **Run the Script**:
   Execute the script in your Python environment:
   ```bash
   python script_name.py
   ```

2. **Expected Output**:
   - The script will connect to the drone and fetch IMU data.
   - Data will be scaled using the scaler and passed to the MLP model for prediction.
   - When the predicted value is non-zero, the output will be displayed, and the script will terminate.

---

## Script Overview

1. **Model and Scaler Loading**:
   ```python
   def load_mlp_model(model_name, scalar_name):
       # Loads the MLP model and scaler from joblib files.
   ```

2. **Drone Connection**:
   ```python
   await drone.connect(system_address="udp://:14540")
   ```

3. **IMU Data Fetching**:
   ```python
   async for imu in telemetry.imu():
       # Fetch gyroscope, accelerometer, and magnetometer data.
   ```

4. **Data Processing**:
   ```python
   input_data = np.array([[accel_data[0], accel_data[1], ...]])
   scaled_data = scalar.transform(input_data)
   predicted_value = mlp_model.predict(scaled_data)
   ```

---

## Requirements.txt Example
Here’s an example of what your `requirements.txt` should include:
```text
asyncio
mavsdk
joblib
numpy
```

---

## License
Include any licensing information here.

---

