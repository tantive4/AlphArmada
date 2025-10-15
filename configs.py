import torch

class Config:
    # Game
    MAX_SHIPS = 6
    MAX_SQUADS = 9
    MAX_COMMAND_STACK = 3
    MAX_DEFENSE_TOKENS = 6 
    MAX_SQUAD_DEFENSE_TOKENS = 2
    GLOBAL_MAX_HULL = 8.0
    GLOBAL_MAX_SHIELDS = 4.0
    GLOBAL_MAX_DICE = 4.0 
    GLOBAL_MAX_SQUAD_VALUE = 4
    GLOBAL_MAX_ENGINEER_VALUE = 4

    # Encoding
    BOARD_RESOLUTION = (32, 16)  # (player_edge width resolution, short_edge height resolution)
    SHIP_ENTITY_FEATURE_SIZE = 90
    SQUAD_ENTITY_FEATURE_SIZE = 32
    RELATION_FEATURE_SIZE = 12
    SCALAR_FEATURE_SIZE = 45

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Training Loop
    ITERATIONS = 4 # self_play & train for ITERATIONS times

    # Self Play
    SELF_PLAY_GAMES = 2 # run SELF_PLAY_GAMES batch self-play games in each iteration
    PARALLEL_PLAY = 2 # run games in batch
    # 4 x 64 x 400 x 0.25 = 12800 states per iteration

    # MCTS
    DEEP_SEARCH_RATIO = 0.25
    MCTS_ITERATION = 20
    MCTS_ITERATION_FAST = 10
    MAX_GAME_STEP = 2000
    TEMPERATURE = 1.25
    EXPLORATION_CONSTANT = 2
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25

    # Replay Buffer
    REPLAY_BUFFER_SIZE = 300000
    REPLAY_BUFFER_DIR = "replay_buffers"

    # Neural Network Training
    TRAINING_STEPS = 4
    BATCH_SIZE = 64
    # 100 x 64 = 6400 training samples per iteration

    # Optimization
    LEARNING_RATE = 0.0001
    L2_LAMBDA = 1e-4
    HULL_LOSS_WEIGHT = 0.05
    SQUAD_LOSS_WEIGHT = 0.05
    GAME_LENGTH_LOSS_WEIGHT = 0.05
    

    # Model Paths
    CHECKPOINT_DIR = "model_checkpoints"