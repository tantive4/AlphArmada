using Armada.Core;
using Armada.Rules;
using System.Numerics;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Objective;

namespace Swarmada.Headless.Runtime;

public sealed record BattleSnapshot(string SchemaId, ShipRuntimeSnapshot[] Ships,
    SquadronRuntimeSnapshot[] Squadrons, CommandStackSnapshot[] Commands,
    DamageDeckSnapshot[] Decks, UpgradeCardSnapshot[] Upgrades, Pose[] Poses,
    bool[] Activated, int[] Owners, long[] Registers, SpatialObjectState[] SpatialObjects,
    int[] VictoryPoints, ObjectiveSnapshot? Objective, RuleFrame[] Frames,
    int Round, long Revision, RandomSnapshot Random, TerminalOutcome? Outcome);

/// <summary>Snapshot-owned mutable map pieces, mines and objective tokens; list index is stable during a match.</summary>
public readonly record struct SpatialObjectState(int Kind, Pose Pose, Vector2 Size, int Owner = 0,
    int SourceEntity = -1, bool Active = true, long Payload = 0, string Id = "", bool Flipped = false, string ShapeId = "");

/// <summary>Complete mutable storage boundary for ported rules. Entity topology stays fixed during a match.</summary>
public sealed class BattleState
{
    public string SchemaId { get; }
    public ShipRuntimeState[] Ships { get; private set; }
    public SquadronRuntimeState[] Squadrons { get; private set; }
    public CommandStackState[] Commands { get; private set; }
    public DamageDeckState[] Decks { get; private set; }
    public UpgradeCardRuntimeState[] Upgrades { get; private set; }
    public Pose[] Poses { get; private set; }
    public bool[] Activated { get; private set; }
    public int[] Owners { get; private set; }
    /// <summary>Named by the schema, never by a handler's instance fields; captures pending tactical context.</summary>
    public long[] Registers { get; private set; }
    public List<SpatialObjectState> SpatialObjects { get; private set; } = new();
    public int[] VictoryPoints { get; private set; } = new int[2];
    public IObjectiveRuntime? Objective { get; set; }
    public List<RuleFrame> Frames { get; private set; } = new();
    public int Round { get; set; } = 1;
    public long Revision { get; internal set; }
    public DeterministicRandom Random;
    public TerminalOutcome? Outcome { get; set; }

    public BattleState(string schemaId, IEnumerable<ShipRuntimeState> ships,
        IEnumerable<SquadronRuntimeState> squadrons, IEnumerable<CommandStackState> commands,
        IEnumerable<DamageDeckState> decks, IEnumerable<UpgradeCardRuntimeState> upgrades,
        int registerCount = 0, ulong seed = 0)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(schemaId);
        if (registerCount < 0) throw new ArgumentOutOfRangeException(nameof(registerCount));
        SchemaId = schemaId;
        Ships = ships.Select(x => x.DeepClone()).ToArray();
        Squadrons = squadrons.Select(x => x.DeepClone()).ToArray();
        Commands = commands.Select(x => x.DeepClone()).ToArray();
        Decks = decks.Select(x => x.DeepClone()).ToArray();
        Upgrades = upgrades.Select(x => x.DeepClone()).ToArray();
        int entityCount = checked(Ships.Length + Squadrons.Length);
        Poses = new Pose[entityCount]; Activated = new bool[entityCount]; Owners = new int[entityCount];
        Registers = new long[registerCount]; Random = new(seed);
    }

    public BattleSnapshot CaptureSnapshot() => new(SchemaId,
        Ships.Select(x => x.CaptureSnapshot()).ToArray(), Squadrons.Select(x => x.CaptureSnapshot()).ToArray(),
        Commands.Select(x => x.CaptureSnapshot()).ToArray(), Decks.Select(x => x.CaptureSnapshot()).ToArray(),
        Upgrades.Select(x => x.CaptureSnapshot()).ToArray(), (Pose[])Poses.Clone(),
        (bool[])Activated.Clone(), (int[])Owners.Clone(), (long[])Registers.Clone(), SpatialObjects.ToArray(),
        (int[])VictoryPoints.Clone(), Objective?.CaptureSnapshot(), Frames.ToArray(),
        Round, Revision, Random.CaptureSnapshot(), Outcome);

    public void RestoreSnapshot(BattleSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        if (snapshot.SchemaId != SchemaId || snapshot.Ships.Length != Ships.Length ||
            snapshot.Squadrons.Length != Squadrons.Length || snapshot.Commands.Length != Commands.Length ||
            snapshot.Decks.Length != Decks.Length || snapshot.Upgrades.Length != Upgrades.Length ||
            snapshot.Poses.Length != Poses.Length || snapshot.Activated.Length != Activated.Length ||
            snapshot.Owners.Length != Owners.Length || snapshot.Registers.Length != Registers.Length ||
            snapshot.VictoryPoints.Length != 2 ||
            snapshot.Round < 1 || snapshot.Revision < 0)
            throw new ArgumentException("Snapshot schema or entity layout does not match this match.", nameof(snapshot));
        // Validate and reconstruct before committing; a rejected snapshot leaves this state unchanged.
        var ships = new ShipRuntimeState[Ships.Length];
        for (int i = 0; i < ships.Length; i++)
        {
            if (snapshot.Ships[i].ShipId != Ships[i].ShipId)
                throw new ArgumentException("Snapshot ship order/identity differs from this match.", nameof(snapshot));
            ships[i] = ShipRuntimeState.FromSnapshot(snapshot.Ships[i]);
        }
        var squads = snapshot.Squadrons.Select(SquadronRuntimeState.FromSnapshot).ToArray();
        var commands = snapshot.Commands.Select(CommandStackState.FromSnapshot).ToArray();
        var decks = snapshot.Decks.Select(DamageDeckState.FromSnapshot).ToArray();
        var upgrades = new UpgradeCardRuntimeState[Upgrades.Length];
        for (int i = 0; i < upgrades.Length; i++)
            upgrades[i] = UpgradeCardRuntimeState.FromSnapshot(snapshot.Upgrades[i], Upgrades[i].Definition);
        var random = new DeterministicRandom(); random.RestoreSnapshot(snapshot.Random);
        var poses = (Pose[])snapshot.Poses.Clone();
        var activated = (bool[])snapshot.Activated.Clone();
        var owners = (int[])snapshot.Owners.Clone();
        var registers = (long[])snapshot.Registers.Clone();
        var spatialObjects = new List<SpatialObjectState>(snapshot.SpatialObjects);
        var victoryPoints = (int[])snapshot.VictoryPoints.Clone();
        var objective = snapshot.Objective is null ? null : ObjectiveRuntime.FromSnapshot(snapshot.Objective);
        var frames = new List<RuleFrame>(snapshot.Frames);
        Ships = ships; Squadrons = squads; Commands = commands; Decks = decks; Upgrades = upgrades;
        Poses = poses; Activated = activated; Owners = owners; Registers = registers; Frames = frames;
        SpatialObjects = spatialObjects; VictoryPoints = victoryPoints;
        Objective = objective;
        Random = random; Round = snapshot.Round; Revision = snapshot.Revision; Outcome = snapshot.Outcome;
    }

    public BattleState DeepClone()
    {
        var clone = (BattleState)MemberwiseClone();
        clone.Ships = new ShipRuntimeState[Ships.Length];
        for (int i = 0; i < Ships.Length; i++) clone.Ships[i] = Ships[i].DeepClone();
        clone.Squadrons = new SquadronRuntimeState[Squadrons.Length];
        for (int i = 0; i < Squadrons.Length; i++) clone.Squadrons[i] = Squadrons[i].DeepClone();
        clone.Commands = new CommandStackState[Commands.Length];
        for (int i = 0; i < Commands.Length; i++) clone.Commands[i] = Commands[i].DeepClone();
        clone.Decks = new DamageDeckState[Decks.Length];
        for (int i = 0; i < Decks.Length; i++) clone.Decks[i] = Decks[i].DeepClone();
        clone.Upgrades = new UpgradeCardRuntimeState[Upgrades.Length];
        for (int i = 0; i < Upgrades.Length; i++) clone.Upgrades[i] = Upgrades[i].DeepClone();
        clone.Poses = (Pose[])Poses.Clone(); clone.Activated = (bool[])Activated.Clone();
        clone.Owners = (int[])Owners.Clone(); clone.Registers = (long[])Registers.Clone();
        clone.SpatialObjects = new(SpatialObjects); clone.VictoryPoints = (int[])VictoryPoints.Clone();
        clone.Frames = new(Frames); clone.Objective = Objective?.DeepClone();
        return clone;
    }

    /// <summary>Reuses allocated physical branch storage. Source must have the same immutable match topology.</summary>
    public void CopyFrom(BattleState source)
    {
        ArgumentNullException.ThrowIfNull(source); if (ReferenceEquals(this, source)) return;
        if (SchemaId != source.SchemaId || Ships.Length != source.Ships.Length || Squadrons.Length != source.Squadrons.Length ||
            Commands.Length != source.Commands.Length || Decks.Length != source.Decks.Length || Upgrades.Length != source.Upgrades.Length ||
            Poses.Length != source.Poses.Length || Activated.Length != source.Activated.Length || Owners.Length != source.Owners.Length || Registers.Length != source.Registers.Length)
            throw new ArgumentException("Branch topology differs from this match.", nameof(source));
        for (int i = 0; i < Ships.Length; i++)
            if (Ships[i].ShipId != source.Ships[i].ShipId) throw new ArgumentException("Branch ship identities differ.", nameof(source));
        for (int i = 0; i < Upgrades.Length; i++)
            if (Upgrades[i].Definition.id != source.Upgrades[i].Definition.id) throw new ArgumentException("Branch upgrade identities differ.", nameof(source));
        for (int i = 0; i < Ships.Length; i++) Ships[i].CopyFrom(source.Ships[i]);
        for (int i = 0; i < Squadrons.Length; i++) Squadrons[i].CopyFrom(source.Squadrons[i]);
        for (int i = 0; i < Commands.Length; i++) Commands[i].CopyFrom(source.Commands[i]);
        for (int i = 0; i < Decks.Length; i++) Decks[i].CopyFrom(source.Decks[i]);
        for (int i = 0; i < Upgrades.Length; i++) Upgrades[i].CopyFrom(source.Upgrades[i]);
        source.Poses.CopyTo(Poses, 0); source.Activated.CopyTo(Activated, 0);
        source.Owners.CopyTo(Owners, 0); source.Registers.CopyTo(Registers, 0); source.VictoryPoints.CopyTo(VictoryPoints, 0);
        SpatialObjects.Clear(); SpatialObjects.AddRange(source.SpatialObjects);
        Frames.Clear(); Frames.AddRange(source.Frames);
        if (source.Objective is null) Objective = null;
        else if (Objective is { } targetObjective && targetObjective.GetType() == source.Objective.GetType()) targetObjective.CopyFrom(source.Objective);
        else Objective = source.Objective.DeepClone();
        Round = source.Round; Revision = source.Revision; Random = source.Random; Outcome = source.Outcome;
    }
    public RuleFrame CurrentFrame => Frames.Count > 0 ? Frames[^1] :
        throw new InvalidOperationException("A nonterminal state has no pending continuation.");
    public RuleFrame PopFrame()
    {
        var frame = CurrentFrame; Frames.RemoveAt(Frames.Count - 1); return frame;
    }
}
