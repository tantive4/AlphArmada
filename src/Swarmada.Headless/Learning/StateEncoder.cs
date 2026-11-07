using System.Numerics;
using Armada.Core;
using Armada.Data;
using Armada.Rules;
using Swarmada.Headless.Catalog;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;
using Swarmada.Headless.Search;
using Swarmada.Headless.Setup;

namespace Swarmada.Headless.Learning;

/// <summary>
/// Read-only, worker-owned reusable observation writer. This is an observation, not an engine
/// recovery snapshot: hidden commands/damage/deck order/RNG are omitted. Call Clone before
/// retaining the result across another Encode call. No decisions are settled and no RNG is used.
/// </summary>
public sealed partial class StateEncoder
{
    private static readonly int[] SpatialKindValues = [0, 1, 2, 3, SetupModule.GravWellKind,
        SetupModule.GravShiftKind, SetupModule.ProximityMineKind, IgnitionAttackGeometry.TargetTokenKind];
    public static IReadOnlyList<int> SpatialKinds { get; } = Array.AsReadOnly(SpatialKindValues);
    public const int GlobalWidth = 256, ShipWidth = 384, SquadWidth = 256, CardWidth = 640,
        PieceWidth = 64, TokenWidth = 16, DieWidth = 24, DamageWidth = 32, CommandWidth = 8,
        PeekWidth = 24, AttackWidth = 192, EffectWidth = 32, ObjectiveWidth = 64, StoreWidth = 16;
    public StateEncoderSchema Schema { get; }
    private readonly StateEncoding output = new();
    private readonly List<EncoderCoverageIssue> issues = [];
    private readonly List<float> effects = [], stores = [];
    private readonly List<int> effectRefs = [], storeRefs = [];
    private readonly List<int> symbolTokens = [];
    private readonly Dictionary<string, int> currentSymbolSequences = new(StringComparer.Ordinal);
    private MatchTopology? topology;
    private int[] placed = [], upgradeOwners = [];
    private float[] shipPrinted = [], squadPrinted = [];
    private Dictionary<string, int> entityIds = new(StringComparer.Ordinal);
    private bool entityIdsHaveSeparators;
    public StateEncoder(FleetData data) : this(StateEncoderSchema.For(data)) { }
    public StateEncoder(StateEncoderSchema schema) => Schema = schema ?? throw new ArgumentNullException(nameof(schema));

    public StateEncoding Encode(GameSession position, SearchKnowledge knowledge, int observer, int actor,
        SearchKnowledge? hypothesisBoundary = null)
    {
        ArgumentNullException.ThrowIfNull(position); ArgumentNullException.ThrowIfNull(knowledge);
        if (observer is not (1 or 2) || knowledge.Observer != observer || actor is < 0 or > 2)
            throw new ArgumentException("Observer, actor or legal-knowledge owner is invalid.");
        knowledge.Validate(position); hypothesisBoundary?.Validate(position);
        var game = position.Context; var state = game.State; var flow = game.Flow;
        PrepareTopology(game); issues.Clear(); effects.Clear(); effectRefs.Clear(); stores.Clear(); storeRefs.Clear(); symbolTokens.Clear(); currentSymbolSequences.Clear();
        output.Observer = observer; output.Actor = actor; output.SchemaFingerprint = Schema.Fingerprint;
        output.Ships = Resize(output.Ships, state.Ships.Length * ShipWidth); output.Squads = Resize(output.Squads, state.Squadrons.Length * SquadWidth);
        output.Cards = Resize(output.Cards, state.Upgrades.Length * CardWidth); output.CardHosts = Resize(output.CardHosts, state.Upgrades.Length);
        output.SquadCommanders = Resize(output.SquadCommanders, state.Squadrons.Length);
        output.Pieces = Resize(output.Pieces, state.SpatialObjects.Count * PieceWidth); output.PieceReferences = Resize(output.PieceReferences, state.SpatialObjects.Count * 5);
        output.Commands = Resize(output.Commands, state.Commands.Sum(x => x.Count) * CommandWidth); output.CommandReferences = Resize(output.CommandReferences, state.Commands.Sum(x => x.Count) * 2);
        output.Damages = Resize(output.Damages, state.Ships.Sum(x => x.DamageCards.Count) * DamageWidth); output.DamageReferences = Resize(output.DamageReferences, state.Ships.Sum(x => x.DamageCards.Count) * 2);
        output.Dice = Resize(output.Dice, flow.Attacks.Sum(x => x.Dice.Length) * DieWidth); output.DieReferences = Resize(output.DieReferences, flow.Attacks.Sum(x => x.Dice.Length) * 2);
        output.Attacks = Resize(output.Attacks, flow.Attacks.Count * AttackWidth); output.AttackReferences = Resize(output.AttackReferences, flow.Attacks.Count * 6);
        int peeks = 0; for (int deck = 0; deck < state.Decks.Length; deck++) peeks += knowledge.KnownDeckPrefix(deck).Count;
        output.KnownDeckPeeks = Resize(output.KnownDeckPeeks, peeks * PeekWidth); output.PeekReferences = Resize(output.PeekReferences, peeks * 2);
        Array.Clear(output.Global); Array.Clear(output.Objective); Array.Copy(shipPrinted, output.Ships, shipPrinted.Length); Array.Copy(squadPrinted, output.Squads, squadPrinted.Length);
        Array.Clear(output.Cards); Array.Clear(output.Pieces); Array.Clear(output.Commands); Array.Clear(output.Damages); Array.Clear(output.Dice); Array.Clear(output.Attacks); Array.Clear(output.KnownDeckPeeks);
        int tokenCount = state.Ships.Sum(x => x.EncoderDefenseSlotCount) + state.Squadrons.Sum(x => x.EncoderDefenseSlotCount);
        output.Tokens = Resize(output.Tokens, tokenCount * TokenWidth); output.TokenReferences = Resize(output.TokenReferences, tokenCount * 3);
        EncodeGlobal(game, observer, actor);
        int commandRow = 0, damageRow = 0;
        for (int i = 0; i < state.Ships.Length; i++)
        {
            var ship = state.Ships[i]; var activation = flow.ShipActivations[i];
            var w = new Writer(output.Ships.AsSpan(i * ShipWidth + 144, ShipWidth - 144));
            Entity(game, i, observer, true, ref w); w.N(ship.CurrentHull / 32f); w.N(ship.MaximumHull / 32f);
            w.N(ship.CurrentHull / (float)Math.Max(1, ship.MaximumHull)); w.N(ship.CurrentSpeed / 4f);
            for (int zone = 0; zone < 6; zone++) { w.N(ship.HullZones.Contains((HullZone)zone) ? ship.GetShields((HullZone)zone) / 10f : 0); w.N(ship.HullZones.Contains((HullZone)zone) ? ship.GetMaximumShields((HullZone)zone) / 10f : 0); }
            var known = knowledge.CommandsFor(position, i); var boundary = hypothesisBoundary?.CommandsFor(position, i);
            for (int slot = 0; slot < known.Length; slot++)
            {
                bool boundaryKnown = boundary is null || slot < boundary.Length && boundary[slot].HasValue;
                if (!boundaryKnown) known[slot] = null;
                var cw = new Writer(output.Commands.AsSpan(commandRow * CommandWidth, CommandWidth));
                // The observation describes legal information, never whether search has
                // allocated/materialized a hypothetical world. Boundary is a visibility
                // gate only; the last feature marks a legally known enemy dial.
                cw.B(true); cw.B(known[slot].HasValue); cw.Hot(4, known[slot].HasValue ? (int)known[slot]!.Value : -1); cw.B(state.Owners[i] == observer); cw.B(known[slot].HasValue && state.Owners[i] != observer);
                output.CommandReferences[commandRow * 2] = i; output.CommandReferences[commandRow * 2 + 1] = slot; commandRow++;
            }
            for (int slot = 0; slot < 4; slot++) w.Hot(6, slot >= known.Length ? 0 : !known[slot].HasValue ? 1 : 2 + (int)known[slot]!.Value);
            int commands = w.Position; w.Skip(8);
            foreach (var command in state.Commands[i].TokenInstances) w.Data[commands + (int)command] += .25f;
            foreach (var command in state.Commands[i].RaidTokens) w.Data[commands + 4 + (int)command] += .25f;
            w.Hot(5, activation.RevealedCommand.HasValue ? 1 + (int)activation.RevealedCommand.Value : 0);
            w.B(activation.RevealedDialAvailable); w.B(activation.ManeuverComplete); w.N(activation.AttacksResolved / 2f);
            for (int zone = 0; zone < 6; zone++) w.B((activation.UsedHullZones & (1 << zone)) != 0);
            for (int command = 0; command < 4; command++) w.B((activation.ResolvedCommands & (1 << command)) != 0);
            w.N(activation.EngineeringPoints / 8f); w.N(activation.CommandedSquadronsRemaining / 8f);
            w.B(activation.NavigateSpeedUsed); w.B(activation.ConcentrateFireDie); w.B(activation.ConcentrateFireReroll);
            int cardOffset = w.Position; w.Skip(22);
            for (int index = 0; index < ship.DamageCards.Count; index++)
            {
                var damage = ship.DamageCards[index]; CriticalDamageCard? visible = damage.FaceUp ? damage.CriticalCard : knowledge.TryKnownDamage(i, index, out var remembered) ? remembered : null;
                if (damage.FaceUp && damage.CriticalCard.HasValue) w.Data[cardOffset + (int)damage.CriticalCard.Value] += .25f;
                var dw = new Writer(output.Damages.AsSpan(damageRow * DamageWidth, DamageWidth)); dw.B(true); dw.B(damage.FaceUp); dw.B(visible.HasValue); dw.Hot(23, visible.HasValue ? 1 + (int)visible.Value : 0);
                output.DamageReferences[damageRow * 2] = i; output.DamageReferences[damageRow * 2 + 1] = index; damageRow++;
            }
            w.N(ship.FacedownDamageCardCount / 32f); w.B(ship.CurrentHull == 0 && !game.IsDestroyed(i));
            Activation(game, i, observer, hypothesisBoundary, ref w);
        }
        for (int i = 0; i < state.Squadrons.Length; i++)
        {
            int entity = state.Ships.Length + i; var squad = state.Squadrons[i];
            var w = new Writer(output.Squads.AsSpan(i * SquadWidth + 160, SquadWidth - 160));
            Entity(game, entity, observer, false, ref w); w.N(squad.CurrentHull / 16f); w.N(squad.CurrentHull / (float)Math.Max(1, game.Topology.Squadrons[i].CatalogEntry.Definition.Hull));
            w.B(squad.Activated); w.B(squad.Commanded); w.B(squad.TemporaryCloak); w.B(squad.TemporaryGrit); w.B(squad.TemporaryRogue);
            w.B(squad.TemporaryIgnoresEngagement); w.B(squad.TemporaryCannotAttack); w.B(squad.EncoderTemporaryAttackOnly); w.B(squad.EncoderMovementPrevented); w.B(squad.TemporaryYavarisBonusDie); w.B(squad.TemporaryCommanded);
            w.N(squad.TemporarySpeedOverride / 5f); w.N(squad.TemporarySnipeValue / 5f); w.B(squad.TemporaryAssault); w.B(squad.TemporarySwarm); w.B(squad.DestructionDeferred);
            output.SquadCommanders[i] = entityIds.GetValueOrDefault(squad.CommandingShipRuntimeId, -1);
            if (flow.SquadronProgress is { } p && p.Entity == entity)
            { w.B(true); w.B(p.Moved); w.B(p.Attacked); w.B(p.Commanded); w.B(p.CanMoveAndAttack); w.B(p.WolffeGrantAdded); w.B(p.ActivationCommitted); w.B(p.Standalone); w.B(p.PreserveActivation); w.N(p.MaximumMoveDistance / 1000); w.N(p.Stage / 16f); w.B(p.PendingMove.HasValue); Point(game, observer, p.PendingMove ?? Vector2.Zero, ref w); }
        }
        EncodeCards(game, observer, hypothesisBoundary);
        EncodePieces(game, observer);
        int tokenRow = 0; var currentAttack = flow.Attacks.LastOrDefault();
        for (int i = 0; i < state.Ships.Length; i++) tokenRow += state.Ships[i].WriteEncoderDefenseSlots(output.Tokens.AsSpan(tokenRow * TokenWidth), output.TokenReferences.AsSpan(tokenRow * 3), i, currentAttack);
        for (int i = 0; i < state.Squadrons.Length; i++) tokenRow += state.Squadrons[i].WriteEncoderDefenseSlots(output.Tokens.AsSpan(tokenRow * TokenWidth), output.TokenReferences.AsSpan(tokenRow * 3), state.Ships.Length + i, currentAttack);
        int peekRow = 0;
        for (int deck = 0; deck < state.Decks.Length; deck++)
        {
            var prefix = knowledge.KnownDeckPrefix(deck);
            for (int slot = 0; slot < prefix.Count; slot++)
            {
                bool visible = true;
                var w = new Writer(output.KnownDeckPeeks.AsSpan(peekRow * PeekWidth, PeekWidth)); w.B(visible); w.Hot(22, visible ? (int)prefix[slot] : -1);
                output.PeekReferences[peekRow * 2] = deck + 1 == observer ? 1 : 2; output.PeekReferences[peekRow * 2 + 1] = slot; peekRow++;
            }
        }
        EncodeAttacks(game, observer); EncodeEffects(game, observer, hypothesisBoundary);
        output.Effects = CopyList(effects, output.Effects); output.EffectReferences = CopyList(effectRefs, output.EffectReferences); output.Stores = CopyList(stores, output.Stores); output.StoreReferences = CopyList(storeRefs, output.StoreReferences);
        output.SymbolTokens = CopyList(symbolTokens, output.SymbolTokens);
        output.CoverageIssues = issues.Distinct().ToArray(); output.RefreshTensors(); return output;
    }

    private void PrepareTopology(GameContext game)
    {
        if (ReferenceEquals(topology, game.Topology)) return;
        topology = game.Topology; placed = Enumerable.Range(0, topology.EntityCount).Select(i => topology.Registers[$"entity.{i}.placed"]).ToArray();
        upgradeOwners = Enumerable.Range(0, game.State.Upgrades.Length).Select(topology.Registers.UpgradeOwner).ToArray();
        entityIds = topology.Ships.Select(x => (x.RuntimeId, x.EntityIndex)).Concat(topology.Squadrons.Select(x => (x.RuntimeId, x.EntityIndex))).ToDictionary(x => x.RuntimeId, x => x.EntityIndex, StringComparer.Ordinal);
        entityIdsHaveSeparators = entityIds.Keys.All(id => id.Contains('|'));
        shipPrinted = new float[game.State.Ships.Length * ShipWidth]; squadPrinted = new float[game.State.Squadrons.Length * SquadWidth];
        for (int i = 0; i < game.State.Ships.Length; i++)
        {
            var d = topology.Ships[i].CatalogEntry.Definition; var w = new Writer(shipPrinted.AsSpan(i * ShipWidth, 144)); w.Hot(68, Schema.Ships.Index(d.ShipId)); w.Hot(4, (int)d.Faction); w.N(d.Points / 400f); w.N(d.Hull / 32f); w.N(d.Command / 4f); w.N(d.Engineering / 8f); w.N(d.Squadron / 8f); w.N(d.GameplayFootprint.X / 1000); w.N(d.GameplayFootprint.Y / 1000);
            for (int zone = 0; zone < 6; zone++) { w.B(d.HasHullZone((HullZone)zone)); var b = d.GetBattery((HullZone)zone); w.N(b.RedDice / 10f); w.N(b.BlueDice / 10f); w.N(b.BlackDice / 10f); }
            w.N(d.AntiSquadronArmament.RedDice / 10f); w.N(d.AntiSquadronArmament.BlueDice / 10f); w.N(d.AntiSquadronArmament.BlackDice / 10f);
            for (int speed = 1; speed <= 4; speed++) { var row = d.GetManeuverRow(speed); w.B(row.IsAvailable); for (int joint = 0; joint < 4; joint++) w.N(row.GetYaw(joint) / 2f); }
            shipPrinted[i * ShipWidth + 382] = topology.Ships[i].IsStation ? 1 : 0;
            shipPrinted[i * ShipWidth + 383] = topology.Ships[i].IsScenarioUnit ? 1 : 0;
        }
        for (int i = 0; i < game.State.Squadrons.Length; i++)
        {
            var d = topology.Squadrons[i].CatalogEntry.Definition; var w = new Writer(squadPrinted.AsSpan(i * SquadWidth, 160)); w.Hot(109, Schema.Squadrons.Index(d.SquadronId)); w.Hot(4, (int)d.Faction); w.N(d.Points / 100f); w.N(d.Hull / 16f); w.N(d.Speed / 5f); w.N(d.GameplayFootprint.X / 1000); w.B(d.IsUnique); w.N(d.AntiShipArmament.RedDice / 10f); w.N(d.AntiShipArmament.BlueDice / 10f); w.N(d.AntiShipArmament.BlackDice / 10f); w.N(d.AntiSquadronArmament.RedDice / 10f); w.N(d.AntiSquadronArmament.BlueDice / 10f); w.N(d.AntiSquadronArmament.BlackDice / 10f);
            int keywords = w.Position; w.Skip(25); foreach (var keyword in d.Keywords) { var parts = keyword.Split(' ', StringSplitOptions.RemoveEmptyEntries); int level = parts.Length > 1 && int.TryParse(parts[^1], out int value) ? value : 1; w.Data[keywords + Schema.Keywords.Index(parts[0].ToLowerInvariant())] = level / 5f; }
        }
    }

    private static T[] Resize<T>(T[] array, int count) => array.Length == count ? array : new T[count];
    private static T[] CopyList<T>(List<T> source, T[] destination) { destination = Resize(destination, source.Count); source.CopyTo(destination); return destination; }
    private static void Point(GameContext game, int observer, Vector2 point, ref Writer writer) { float sign = observer == 1 ? 1 : -1; writer.N(sign * point.X * 2 / game.Topology.BoardSizeMillimetres.X); writer.N(sign * point.Y * 2 / game.Topology.BoardSizeMillimetres.Y); }
    private static void Heading(int observer, float yaw, ref Writer writer) { float radians = (yaw + (observer == 2 ? 180 : 0)) * MathF.PI / 180; writer.N(MathF.Sin(radians)); writer.N(MathF.Cos(radians)); }
    private void Entity(GameContext game, int entity, int observer, bool heading, ref Writer w)
    {
        bool deployed = game.State.Registers[placed[entity]] != 0 && !game.IsReserved(entity); var attack = game.Flow.Attacks.LastOrDefault();
        w.B(true); w.B(game.State.Owners[entity] == observer); w.B(game.IsDestroyed(entity)); w.B(game.IsReserved(entity)); w.B(deployed); w.B(game.State.Activated[entity]); w.B(game.Flow.ActiveShip == entity); w.B(game.Flow.ActiveSquadron == entity); w.B(attack?.Attacker == entity); w.B(attack?.Defender == entity);
        Point(game, observer, deployed ? game.State.Poses[entity].Position : Vector2.Zero, ref w); if (heading) { if (deployed) Heading(observer, game.State.Poses[entity].YawDegrees, ref w); else { w.N(0); w.N(0); } }
        if (game.State.Objective is OfficialObjectiveState objective) { w.N(objective.EntityTokens[entity] / 4f); w.B(objective.ObjectiveShips[entity]); } else { w.N(0); w.N(0); }
    }
    private ref struct Writer(Span<float> data)
    {
        public Span<float> Data = data;
        public int Position;
        public void N(float value) { if (!float.IsFinite(value)) throw new InvalidOperationException("Non-finite encoder feature."); if ((uint)Position >= (uint)Data.Length) throw new InvalidOperationException("Encoder schema row overflow."); Data[Position++] = value; }
        public void B(bool value) => N(value ? 1 : 0);
        public void Skip(int count) { Position = checked(Position + count); if (Position > Data.Length) throw new InvalidOperationException("Encoder schema row overflow."); }
        public void Hot(int width, int value) { if (value >= width || value < -1) throw new InvalidOperationException($"Encoder category {value} exceeds versioned block width {width} at column {Position}."); if (Position + width > Data.Length) throw new InvalidOperationException("Encoder schema row overflow."); if (value >= 0) Data[Position + value] = 1; Skip(width); }
    }
}
