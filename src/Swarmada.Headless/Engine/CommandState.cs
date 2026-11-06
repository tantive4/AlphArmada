using Armada.Rules;

namespace Swarmada.Headless.Engine;

public sealed partial class ShipActivationState
{
    public bool ConcentrateFireDie { get; set; }
    public bool ConcentrateFireReroll { get; set; }
    public bool ConcentrateFireRavagerChoice { get; set; }
    public int NavigateSpeedLimit { get; set; }
    public bool NavigateSpeedUsed { get; set; }
    public bool RevealedDialAvailable { get; set; }
    public bool CommandResolutionCompleted { get; set; }
    public ShipCommand? PendingCommand { get; set; }
    public int ResolvedDialCommands { get; set; }
    public List<AdditionalCommandDial> AdditionalCommandDials { get; set; } = [];
    public int CommandRevealStep { get; set; } = 5;
    public bool RapidLaunchReleaseChecked { get; set; }
    public int BailDialSelection { get; set; }
    public bool ThrawnDialQueued { get; set; }
    public bool AdditionalDialActive { get; set; }
    public bool CurrentCommandResolved { get; set; }
    public ShipCommand? CurrentResolvedCommand { get; set; }
    public bool CommandCompletionPending { get; set; }
    public int CommandCompletionStep { get; set; }
    public GamePhase CommandCompletionReturnPhase { get; set; }
    public bool VantoOffered { get; set; }
    public List<int> CommandedSquadrons { get; set; } = [];
    public int SquadronCommandPreludeStep { get; set; } = 4;
    public bool RaidInterferenceResolved { get; set; }
    public bool NativeDialConsumedBeforePrelude { get; set; }
    public int DeferredSquadronActivations { get; set; }
    public CommandSpendMode DeferredSquadronSpendMode { get; set; }
    public int BeckCommands { get; set; }
    public int UncommittedRelayKeyword { get; set; } = -1;
    public int UncommittedRelayTitle { get; set; } = -1;
    public int UncommittedRelayTarget { get; set; } = -1;
    public bool UncommittedMatchstick { get; set; }
    public bool ShieldsToMaximumChecked { get; set; }
    partial void CopyCommandStateFrom(ShipActivationState other)
    {
        ConcentrateFireDie = other.ConcentrateFireDie; ConcentrateFireReroll = other.ConcentrateFireReroll;
        ConcentrateFireRavagerChoice = other.ConcentrateFireRavagerChoice;
        NavigateSpeedLimit = other.NavigateSpeedLimit; NavigateSpeedUsed = other.NavigateSpeedUsed;
        RevealedDialAvailable = other.RevealedDialAvailable; CommandResolutionCompleted = other.CommandResolutionCompleted;
        PendingCommand = other.PendingCommand; ResolvedDialCommands = other.ResolvedDialCommands;
        AdditionalCommandDials.Clear(); AdditionalCommandDials.AddRange(other.AdditionalCommandDials); CommandRevealStep = other.CommandRevealStep;
        RapidLaunchReleaseChecked = other.RapidLaunchReleaseChecked;
        BailDialSelection = other.BailDialSelection; ThrawnDialQueued = other.ThrawnDialQueued;
        AdditionalDialActive = other.AdditionalDialActive; CurrentCommandResolved = other.CurrentCommandResolved;
        CurrentResolvedCommand = other.CurrentResolvedCommand;
        CommandCompletionPending = other.CommandCompletionPending; CommandCompletionStep = other.CommandCompletionStep;
        CommandCompletionReturnPhase = other.CommandCompletionReturnPhase; VantoOffered = other.VantoOffered;
        CommandedSquadrons.Clear(); CommandedSquadrons.AddRange(other.CommandedSquadrons);
        SquadronCommandPreludeStep = other.SquadronCommandPreludeStep;
        RaidInterferenceResolved = other.RaidInterferenceResolved;
        NativeDialConsumedBeforePrelude = other.NativeDialConsumedBeforePrelude;
        DeferredSquadronActivations = other.DeferredSquadronActivations; DeferredSquadronSpendMode = other.DeferredSquadronSpendMode;
        BeckCommands = other.BeckCommands;
        UncommittedRelayKeyword = other.UncommittedRelayKeyword; UncommittedRelayTitle = other.UncommittedRelayTitle;
        UncommittedRelayTarget = other.UncommittedRelayTarget; UncommittedMatchstick = other.UncommittedMatchstick;
        ShieldsToMaximumChecked = other.ShieldsToMaximumChecked;
    }
}

public readonly record struct AdditionalCommandDial(ShipCommand Command, string Source);

public sealed partial class GameFlowState
{
    public GamePhase CommandReturnPhase { get; set; } = GamePhase.SelectAttack;
    public CommandSpendMode EffectiveCommandSpendMode { get; set; }
    partial void CopyCommandFlowStateFrom(GameFlowState other)
    {
        CommandReturnPhase = other.CommandReturnPhase;
        EffectiveCommandSpendMode = other.EffectiveCommandSpendMode;
    }
}
