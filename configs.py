import torch

class Config:
    # Game
    MAX_SHIPS = 10
    # MAX_SQUADS = 16
    MAX_SQUADS = 0
    # MAX_COMMAND_STACK = 3
    MAX_COMMAND_STACK = 0
    MAX_DEFENSE_TOKENS = 4
    MAX_SQUAD_DEFENSE_TOKENS = 2
    GLOBAL_MAX_HULL = 8.0
    GLOBAL_MAX_SHIELDS = 4.0
    GLOBAL_MAX_DICE = 4.0
    GLOBAL_MAX_SQUAD_VALUE = 4
    GLOBAL_MAX_ENGINEER_VALUE = 4

    # Encoding
    BOARD_RESOLUTION = (64, 128)  # (short_edge height resolution, player_edge width resolution)
    SHIP_ENTITY_FEATURE_SIZE = 128
    DEF_TOKEN_FEATURE_SIZE = 8
    SQUAD_ENTITY_FEATURE_SIZE = 30
    SCALAR_FEATURE_SIZE = 48
    MAX_ACTION_SPACE = 947

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    GPU_INPUT_BATCH_SIZE = 128  # for neural network inference batch size

    # Training Loop
    ITERATIONS = 64 # self_play & train for ITERATIONS times

    # Self Play
    PARALLEL_DIVERSE_FACTOR = 8 # run games in batch
    PARALLEL_SAME_GAME = 16 # same geometry game setup
    PARALLEL_PLAY = PARALLEL_DIVERSE_FACTOR * PARALLEL_SAME_GAME
    # 1 * 128 * 400 * 0.25 = 12800 states are created

    # MCTS
    DEEP_SEARCH_RATIO = 0.25
    MCTS_ITERATION = 200
    MCTS_ITERATION_FAST = 50
    MAX_GAME_STEP = 2000
    TEMPERATURE = 1.25
    EXPLORATION_CONSTANT = 2
    DIRICHLET_ALPHA_SCALE = 10.0
    DIRICHLET_EPSILON = 0.25

    # Replay Buffer
    REPLAY_BUFFER_SIZE = 16000 # safe buffer for ONE batch games
    

    # Neural Network Training
    EPOCHS = 100
    BATCH_SIZE = 128
    # train on 128 * 100 = 12800 states per iteration


    # Optimization
    LEARNING_RATE = 0.0001
    L2_LAMBDA = 1e-4
    HULL_LOSS_WEIGHT = 0.05
    SQUAD_LOSS_WEIGHT = 0.05
    GAME_LENGTH_LOSS_WEIGHT = 0.05
    

    # Model Paths
    CHECKPOINT_DIR = f"model_checkpoints"
    REPLAY_BUFFER_DIR = f"replay_buffers"