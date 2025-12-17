import torch

class Config:
    # Game
    MAX_SHIPS = 10
    MAX_SQUADS = 16
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
    SHIP_ENTITY_FEATURE_SIZE = 108
    SQUAD_ENTITY_FEATURE_SIZE = 30
    RELATION_FEATURE_SIZE = 12
    SCALAR_FEATURE_SIZE = 45

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    GPU_INPUT_BATCH_SIZE = 128  # for neural network inference batch size

    # Training Loop
    ITERATIONS = 64 # self_play & train for ITERATIONS times

    # Self Play
    SELF_PLAY_GAMES = 1 # run SELF_PLAY_GAMES batch self-play games in each iteration
    PARALLEL_DIVERSE_FACTOR = 32 # run games in batch
    PARALLEL_SAME_GAME = 16 # same geometry game setup
    PARALLEL_PLAY = PARALLEL_DIVERSE_FACTOR * PARALLEL_SAME_GAME
    # 1 * 512 * 400 * 0.25 = 51200 states are created

    # MCTS
    DEEP_SEARCH_RATIO = 0.25
    MCTS_ITERATION = 200
    MCTS_ITERATION_FAST = 50
    MAX_GAME_STEP = 2000
    TEMPERATURE = 1.25
    EXPLORATION_CONSTANT = 2
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25

    # Replay Buffer
    REPLAY_BUFFER_SIZE = 128000  # 10 iteration to take for full buffer replacement
    REPLAY_BUFFER_DIR = "replay_buffers"

    # Neural Network Training
    EPOCHS = 400
    BATCH_SIZE = 128
    # train on 128 * 400 = 51200 states per iteration


    # Optimization
    LEARNING_RATE = 0.0001
    L2_LAMBDA = 1e-4
    HULL_LOSS_WEIGHT = 0.05
    SQUAD_LOSS_WEIGHT = 0.05
    GAME_LENGTH_LOSS_WEIGHT = 0.05
    

    # Model Paths
    CHECKPOINT_DIR = "model_checkpoints"