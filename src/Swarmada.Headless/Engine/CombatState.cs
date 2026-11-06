using Armada.Core;
using Armada.Rules;

namespace Swarmada.Headless.Engine;

public enum CombatWindow { BeforeRoll, AttackEffects, DefenseEffects, CriticalEffects, BeforeDamage, AfterAttack }

public sealed partial class AttackState
{
    public GamePhase ResumePhase { get; set; } = GamePhase.SelectAttack;
    public CombatWindow Window { get; set; }
    public bool EnvironmentalDamage { get; set; }
    public bool IsSalvoAttack { get; set; }
    public bool SnipeAttack { get; set; }
    public bool IgnitionAttack { get; set; }
    public int MineSpatialIndex { get; set; } = -1;
    public bool ExtremeRange { get; set; }
    public RangeBand DiceRange { get; set; }
    public bool IgnoreShields { get; set; }
    public bool NeedsDamageZone { get; set; }
    public bool Rolled { get; set; }
    public bool PreRollComplete { get; set; }
    public int MandatoryRemovals { get; set; }
    public int[] PendingRollIndices { get; set; } = [];
    public GamePhase AfterRollPhase { get; set; } = GamePhase.ModifyAttack;
    public DefenseTokenInstance? SpentEvade { get; set; }
    public bool EvadeWasDiscarded { get; set; }
    public int EvadeBaseCount { get; set; } = 1;
    public bool EvadeCancels { get; set; }
    public int PendingHullCards { get; set; }
    public int HullCardsDealt { get; set; }
    public bool FirstHullCardFaceup { get; set; }
    public CriticalDamageCard? PendingDamageCard { get; set; }
    public CriticalDamageCard? IncomingDamageCard { get; set; }
    public int[] CriticalChosenZones { get; set; } = [];
    public bool DamageStarted { get; set; }
    public int AppliedDamage { get; set; }
    public bool ResponseStarted { get; set; }
    public bool AfterAttackWindowEntered { get; set; }
    public IntelOfficerTokenTarget? IntelOfficerTarget { get; set; }
    public string DamageRule { get; set; } = "";
    public GamePhase? ParentResumePhase { get; set; }
    public string EffectRule { get; set; } = "";
    public string EffectActionId { get; set; } = "";
    public int EffectStage { get; set; }
    public int EffectOwner { get; set; }
    public int EffectMinimum { get; set; }
    public int EffectMaximum { get; set; }
    public int EffectValue { get; set; }
    public int[] EffectIndices { get; set; } = [];
    public int[] EffectEligibleIndices { get; set; } = [];
    public Dictionary<string, long> EffectFlags { get; set; } = new(StringComparer.Ordinal);

    partial void CloneCombatFieldsFrom(AttackState other)
    {
        PendingRollIndices = other.PendingRollIndices.ToArray();
        CriticalChosenZones = other.CriticalChosenZones.ToArray();
        EffectIndices = other.EffectIndices.ToArray();
        EffectEligibleIndices = other.EffectEligibleIndices.ToArray();
        EffectFlags = new(other.EffectFlags, StringComparer.Ordinal);
    }

    /// <summary>Restores a search branch into reusable storage without sharing mutable arrays.</summary>
    public void CopyFrom(AttackState other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (ReferenceEquals(this, other)) return;
        Attacker = other.Attacker; Defender = other.Defender; FromZone = other.FromZone; ToZone = other.ToZone;
        Range = other.Range; Obstructed = other.Obstructed; Counter = other.Counter; Salvo = other.Salvo;
        Brace = other.Brace; Scatter = other.Scatter; Contain = other.Contain; Redirect = other.Redirect;
        Evade = other.Evade; CriticalResolved = other.CriticalResolved; Damage = other.Damage;
        RedDice = other.RedDice; BlueDice = other.BlueDice; BlackDice = other.BlackDice;
        Dice = CopyArray(Dice, other.Dice); LockedTokens = CopyArray(LockedTokens, other.LockedTokens);
        SpentTokenTypes = CopyArray(SpentTokenTypes, other.SpentTokenTypes);
        RedirectAllocations = CopyArray(RedirectAllocations, other.RedirectAllocations);
        ResumePhase = other.ResumePhase; Window = other.Window; EnvironmentalDamage = other.EnvironmentalDamage;
        IsSalvoAttack = other.IsSalvoAttack; SnipeAttack = other.SnipeAttack; IgnitionAttack = other.IgnitionAttack; MineSpatialIndex = other.MineSpatialIndex;
        ExtremeRange = other.ExtremeRange; DiceRange = other.DiceRange; IgnoreShields = other.IgnoreShields;
        NeedsDamageZone = other.NeedsDamageZone; Rolled = other.Rolled; PreRollComplete = other.PreRollComplete;
        MandatoryRemovals = other.MandatoryRemovals; AfterRollPhase = other.AfterRollPhase;
        SpentEvade = other.SpentEvade; EvadeWasDiscarded = other.EvadeWasDiscarded;
        EvadeBaseCount = other.EvadeBaseCount; EvadeCancels = other.EvadeCancels;
        PendingHullCards = other.PendingHullCards; HullCardsDealt = other.HullCardsDealt;
        FirstHullCardFaceup = other.FirstHullCardFaceup; PendingDamageCard = other.PendingDamageCard; IncomingDamageCard = other.IncomingDamageCard;
        DamageStarted = other.DamageStarted; AppliedDamage = other.AppliedDamage; ResponseStarted = other.ResponseStarted;
        AfterAttackWindowEntered = other.AfterAttackWindowEntered; IntelOfficerTarget = other.IntelOfficerTarget;
        DamageRule = other.DamageRule; ParentResumePhase = other.ParentResumePhase;
        EffectRule = other.EffectRule; EffectActionId = other.EffectActionId; EffectStage = other.EffectStage;
        EffectOwner = other.EffectOwner; EffectMinimum = other.EffectMinimum; EffectMaximum = other.EffectMaximum;
        EffectValue = other.EffectValue;
        PendingRollIndices = CopyArray(PendingRollIndices, other.PendingRollIndices);
        CriticalChosenZones = CopyArray(CriticalChosenZones, other.CriticalChosenZones);
        EffectIndices = CopyArray(EffectIndices, other.EffectIndices);
        EffectEligibleIndices = CopyArray(EffectEligibleIndices, other.EffectEligibleIndices);
        EffectFlags.Clear(); foreach (var entry in other.EffectFlags) EffectFlags.Add(entry.Key, entry.Value);
    }
    private static T[] CopyArray<T>(T[] storage, T[] value)
    {
        if (storage.Length != value.Length) return value.ToArray();
        Array.Copy(value, storage, value.Length); return storage;
    }
}

/// <summary>Stateless source-rule integration. Mutable card/window state belongs to GameContext.
/// Unknown active upgrade IDs are rejected before an attack rather than silently skipped.</summary>
public interface ICombatEffectHooks
{
    bool SupportsUpgrade(string id);
    IEnumerable<GameActionOption> GetActions(GameContext game, AttackState attack, CombatWindow window);
    bool TryApply(GameContext game, AttackState attack, CombatWindow window, GameAction action);
    void EnterWindow(GameContext game, AttackState attack, CombatWindow window);
    void AfterDiceChance(GameContext game, AttackState attack) { }
    bool HasPendingChoice(GameContext game, AttackState attack) => false;
    int? DecisionPlayer(GameContext game, AttackState attack) => null;
    bool TryFinishWindow(GameContext game, AttackState attack, CombatWindow window) => false;
    bool AllowsStandardCritical(GameContext game, AttackState attack) => true;
}
