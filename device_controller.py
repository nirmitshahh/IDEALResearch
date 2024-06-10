import serial
import time
import numpy as np

class GaitDeviceController:
    """
    Controls the biomechanical gait rehabilitation device.
    Communicates with device via serial port.
    """
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.connection = None
        self.is_connected = False
        
    def connect(self):
        """Establish connection to device"""
        try:
            self.connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for connection to stabilize
            self.is_connected = True
            print(f"Connected to device on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Close connection to device"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            self.is_connected = False
            print("Disconnected from device")
    
    def send_command(self, command):
        """Send command to device"""
        if not self.is_connected:
            print("Device not connected")
            return False
        
        try:
            self.connection.write(f"{command}\n".encode())
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def set_actuator_position(self, actuator_id, position):
        """Set position of specific actuator"""
        command = f"SET_ACTUATOR {actuator_id} {position}"
        return self.send_command(command)
    
    def set_gait_pattern(self, pattern_data):
        """Set complete gait pattern for device"""
        # pattern_data should be array of actuator positions over time
        for i, positions in enumerate(pattern_data):
            for actuator_id, pos in enumerate(positions):
                self.set_actuator_position(actuator_id, pos)
            time.sleep(0.01)  # Small delay between steps
    
    def read_sensor_data(self):
        """Read sensor data from device"""
        if not self.is_connected:
            return None
        
        try:
            if self.connection.in_waiting > 0:
                data = self.connection.readline().decode().strip()
                return data
            return None
        except Exception as e:
            print(f"Error reading data: {e}")
            return None
    
    def start_rehabilitation_session(self):
        """Start a rehabilitation session"""
        return self.send_command("START_SESSION")
    
    def stop_rehabilitation_session(self):
        """Stop current rehabilitation session"""
        return self.send_command("STOP_SESSION")
    
    def get_status(self):
        """Get current device status"""
        self.send_command("STATUS")
        time.sleep(0.1)
        return self.read_sensor_data()

