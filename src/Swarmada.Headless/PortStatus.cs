namespace Swarmada.Headless;

/// <summary>Executable coverage is deliberately separate from recovered-source/content coverage.</summary>
public static class PortStatus
{
    public const string UpstreamVersion = "0.2.7";
    public static bool HeadlessMatchReady => true;
    public static bool BranchReplayReady => true;
    public static bool StructuredEncoderReady => true;
    public static int ExecutableOfficialObjectiveCount => Engine.OfficialObjectives.SupportedIds.Length;
    // Conservative official-parity flags include deferred deck behavior and validation limits.
    public static bool FullGameReady => false;
    public static bool TrainingReady => false;
    public static string[] RemainingGameplayWork =>
    [
        "All 36 objective handlers remain available. The default static-map training variant disables relocation mechanics; see README.md for modified objective behavior.",
        "All 13 physical obstacle silhouettes use recovered concave mesh outlines and support both faces. Small source-to-physical calibration differences remain; see README.md.",
        "The user's reduced-endpoint ram, retryable Fleet Ambush setup, persistent initial exogorth host, and no late Rieekan rulings are implemented.",
        "Damage-deck initialization and exhaustion behavior are intentionally unchanged at the user's request: matchId-derived order and Structural Damage when empty.",
        "Card registration and focused regressions do not establish exhaustive correctness for every card interaction.",
        "Discrete action IDs and candidates are caller-owned. The engine masks supplied actions and applies the selected ID without generating, ranking or substituting moves.",
        "Obstacle and map-token relocation defaults off. Initial setup and newly placed mine/ignition markers remain separate spatial choices; no universal candidate-count cap is claimed.",
        "Obstacle minimum progress and fallback preferences are caller policy under an explicit rules variation; continuous destination optimization and its proof cache are removed.",
        "Native MCTS, structured visible-state/spatial/policy tensors, batched random-model validation and lossless replay transport are implemented. BigDeep, optimizer/trainer and the complete per-card policy registry remain unfinished.",
        "Pre-reveal command-dependent officers, Isard and the user-selected Draven private look use first-dependency UCT and remembered seen dials. Thrawn/Trench card inventories use canonical non-neural UCT hypotheses fixed for each continuation; Bail uses public token choices. Hypothesis nodes are not neural requests or policy targets. Encoder coverage errors remain explicit; full unrestricted self-play readiness is not claimed."
    ];
}
