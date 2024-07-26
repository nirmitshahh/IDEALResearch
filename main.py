"""
Main script to run gait rehabilitation device with trajectory optimization
and autonomous navigation capabilities.
"""

import numpy as np
import pandas as pd  # moved import to top
from gait_analyzer import GaitAnalyzer
from trajectory_optimizer import TrajectoryOptimizer
from device_controller import GaitDeviceController
from navigation_rl import NavigationAgent, TerrainEnvironment

def generate_sample_gait_data():
    """Generate sample gait data for testing"""
    # create fake data if we don't have real data
    time_points = np.linspace(0, 2, 200)  # 2 seconds of data
    acceleration = np.sin(2 * np.pi * time_points) + 0.3 * np.random.randn(200)
    velocity = np.cumsum(acceleration) * 0.01  # integrate to get velocity
    return pd.DataFrame({
        'time': time_points,
        'acceleration': acceleration,
        'velocity': velocity
    })

def main():
    print("Initializing Gait Rehabilitation System...")
    
    # Initialize components
    analyzer = GaitAnalyzer(sampling_rate=100)
    device = GaitDeviceController()
    # navigation stuff added
    
    # Load or generate gait data
    try:
        gait_data = analyzer.load_data('gait_data.csv')
    except FileNotFoundError:
        print("No gait data file found, using sample data")
        gait_data = generate_sample_gait_data()
    
    # Analyze gait patterns
    print("Analyzing gait patterns...")
    steps = analyzer.detect_steps(gait_data['acceleration'].values)
    step_lengths = analyzer.calculate_step_length(steps, gait_data['velocity'].values)
    
    print(f"Detected {len(steps)} steps")
    print(f"Average step length: {np.mean(step_lengths):.3f} m")
    
    # Optimize trajectory
    print("\nOptimizing walking trajectory...")
    # example target path - in real use this comes from patient's goal
    target_path = np.array([[0, 0], [1, 0.5], [2, 1], [3, 1.5], [4, 2]])
    optimizer = TrajectoryOptimizer(target_path, stability_threshold=0.7)
    optimized_path = optimizer.optimize()
    
    mse_reduction = optimizer.calculate_mse_reduction()
    print(f"MSE reduction: {mse_reduction:.1f}%")
    optimizer.visualize_paths()  # saves plot to file
    
    # Connect to device
    if device.connect():
        print("Device connected, starting rehabilitation session...")
        device.start_rehabilitation_session()
        
        # Apply optimized trajectory to device
        # need to convert path coordinates to actuator positions
        # simplified scaling - real version would be more complex
        pattern_data = optimized_path * 10  # scale to actuator range
        device.set_gait_pattern(pattern_data)
        
        print("Rehabilitation session complete")
        device.stop_rehabilitation_session()
        device.disconnect()
    
    # Test navigation agent
    print("\nTesting autonomous navigation...")
    env = TerrainEnvironment(size=(10, 10))
    agent = NavigationAgent()
    
    for episode in range(100):
        state = env.reset()
        total_reward = 0
        
        for step in range(200):
            action = agent.act(state, training=True)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.replay()
        
        if episode % 20 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    print("Navigation training complete")
    agent.save_model('navigation_model.pkl')

if __name__ == "__main__":
    main()

