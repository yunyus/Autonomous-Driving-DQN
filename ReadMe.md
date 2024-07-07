# Autonomous Driving with DQN in CARLA

This repository contains the implementation of a Deep Q-Network (DQN) algorithm to train an autonomous driving model in CARLA. The goal of the model is to learn how to drive in a lane as fast as possible without any collisions.

## Project Structure

```
autonomous_driving_dqn/
│
├── src/
│   ├── Environment.py          # Defines the CARLA environment
│   ├── Hyperparameters.py      # Contains hyperparameters for training
│   ├── Main.py                 # Script to train the DQN model
│   ├── Model.py                # Defines the DQN model architecture
│   ├── Test.py                 # Script to test the trained DQN model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Setup

1. Clone the repository:

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Ensure you have CARLA installed. Follow the instructions on the [CARLA website](https://carla.org/) to set up CARLA on your system.

## Usage

### Training

To train the DQN model, run:

```
python src/Main.py
```

### Testing

To test the trained DQN model, run:

```
python src/Test.py
```

## Requirements

Make sure to have a CARLA server running before executing the training or testing scripts. You can start the CARLA server using:

```
./CarlaUE4.sh
```

or for Windows:

```
CarlaUE4.exe
```
