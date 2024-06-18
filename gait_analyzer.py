import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Gait pattern analyzer for rehabilitation device
# This reads sensor data and figures out walking patterns

class GaitAnalyzer:
    """
    Analyzes gait patterns from sensor data to identify abnormalities
    and track rehabilitation progress.
    """
    
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        self.gait_cycles = []  # store detected cycles here
        
    def load_data(self, filepath):
        """Load gait data from CSV file"""
        self.data = pd.read_csv(filepath)
        return self.data
    
    def detect_steps(self, acceleration_data):
        """Detect individual steps from acceleration data"""
        # tried a few methods, peak detection works best
        # the 0.5 second distance prevents detecting same step twice
        mean_accel = np.mean(acceleration_data)
        min_distance = int(self.sampling_rate * 0.5)  # half second between steps
        peaks, _ = signal.find_peaks(acceleration_data, 
                                     height=mean_accel,
                                     distance=min_distance)
        return peaks
    
    def calculate_step_length(self, step_indices, velocity_data):
        """Calculate step length from velocity data"""
        step_lengths = []
        for i in range(len(step_indices) - 1):
            start_idx = step_indices[i]
            end_idx = step_indices[i + 1]
            step_length = np.trapz(velocity_data[start_idx:end_idx]) / self.sampling_rate
            step_lengths.append(step_length)
        return np.array(step_lengths)
    
    def analyze_gait_cycle(self, cycle_data):
        """Analyze a single gait cycle"""
        cycle_info = {
            'duration': len(cycle_data) / self.sampling_rate,
            'mean_velocity': np.mean(cycle_data['velocity']),
            'max_acceleration': np.max(cycle_data['acceleration']),
            'stability_score': self._calculate_stability(cycle_data)
        }
        return cycle_info
    
    def _calculate_stability(self, cycle_data):
        """Calculate stability metric based on acceleration variance"""
        # more variance = less stable, so invert it
        accel_variance = np.var(cycle_data['acceleration'])
        # this formula gives higher score for lower variance
        stability = 1.0 / (1.0 + accel_variance)
        return stability
    
    def compare_to_baseline(self, current_data, baseline_data):
        """Compare current gait pattern to baseline"""
        current_stability = self._calculate_stability(current_data)
        baseline_stability = self._calculate_stability(baseline_data)
        
        improvement = ((current_stability - baseline_stability) / baseline_stability) * 100
        return improvement

# fix step detection
