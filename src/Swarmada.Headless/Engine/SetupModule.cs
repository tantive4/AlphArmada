using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Effects;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Interactive source setup: reserves, continuous placement, fixed effects and card choices.</summary>
public sealed partial class SetupModule : IGameplayModule
{
    public static bool SupportsUpgrade(string id) => id is "admiral-ozzel-officer" or "admiral-raddus-commander" or "profundity-title" or
        "corvus-title" or "g7-x-grav-well-projector" or "grav-shift-reroute" or "hyperspace-rings" or "minister-tua-officer" or "flag-bridge" or "liberator-title" or
        "invincible-title" or "radiant-vii-title";
    public const int GravWellKind = 100;
    public const int GravShiftKind = 101;
    public const int ProximityMineKind = 102;
    public bool Handles(GamePhase phase) => phase == GamePhase.Deployment;

    public GameDecision GetDecision(GameContext game)
    {
        if (OfficialObjectives.TryBeginBeforeDeployment(game)) return new(DecisionKind.Forced, 0, game.Flow.Phase, []);
        EnsureInitialized(game);
        while (true)
        {
            if (game.Flow.Phase != GamePhase.Deployment) return new(DecisionKind.Forced, 0, game.Flow.Phase, []);
            int stage = (int)game.Flow.Flags.GetValueOrDefault("setup.engine-stage");
            if (stage is 0 or 2)
            {
                if (TryBeginSetupTiming(game, stage)) return new(DecisionKind.Forced, 0, game.Flow.Phase, []);
                var pending = Current(game);
                if (pending is not null)
                {
                    if (game.IsReserved(pending.Source) || game.IsDestroyed(pending.Source)) { RemoveCurrent(game); continue; }
                    var choices = ChoiceOptions(game, pending);
                    if (choices.Count == 0) { RemoveCurrent(game); continue; }
                    return new(DecisionKind.Player, game.State.Owners[pending.Source], GamePhase.Deployment, choices);
                }
                if (stage == 0) { game.Flow.Flags["setup.engine-stage"] = 1; game.Flow.CurrentPlayer = game.Topology.FirstPlayer; continue; }
                FinishSetup(game);
                return new(DecisionKind.Forced, 0, game.Flow.Phase, Array.Empty<GameActionOption>());
            }
            OfficialDefenseObjectives.ResolveUndeployableSquadrons(game);
            var options = DeploymentOptions(game, game.Flow.CurrentPlayer);
            if (options.Count == 0)
            {
                int other = 3 - game.Flow.CurrentPlayer;
                if (DeploymentOptions(game, other).Count > 0)
                { game.Flow.CurrentPlayer = other; game.Flow.Flags["setup.squadrons-this-turn"] = 0; continue; }
                PreparePostDeployment(game); continue;
            }
            return new(DecisionKind.Player, game.Flow.CurrentPlayer, GamePhase.Deployment, options);
        }
    }

    public void Apply(GameContext game, GameAction action)
    {
        var decision = GetDecision(game);
        if (game.Flow.Phase != GamePhase.Deployment) throw new ArgumentException("Deployment is complete.");
        var option = decision.Actions.FirstOrDefault(x => Matches(x, action));
        if (option is null) throw new ArgumentException("Illegal setup action.", nameof(action));
        int stage = (int)game.Flow.Flags["setup.engine-stage"];
        if (stage == 1)
        {
            bool squadron;
            if (action.AbilityId == "setup.invisible-hand.reserve")
            {
                var reserveCard = game.State.Upgrades[action.Index];
                if (!reserveCard.StoreUnit(RuntimeId(game, action.Target))) throw new ArgumentException("Squadron is already reserved.");
                game.State.Registers[game.Topology.Registers.EntityReserved(action.Target)] = 1;
                game.State.Registers[game.Topology.Registers[$"entity.{action.Target}.reserveUpgradePlusOne"]] = action.Index + 1;
                game.State.Registers[game.Topology.Registers[$"entity.{action.Target}.reserveSourcePlusOne"]] = action.Source + 1;
                squadron = true;
            }
            else { PlaceEntity(game, action.Source, action, redeploy: false); squadron = !game.IsShip(action.Source); }
            if (squadron) game.Flow.Flags["setup.squadrons-this-turn"] = game.Flow.Flags.GetValueOrDefault("setup.squadrons-this-turn") + 1;
            if (!squadron || game.Flow.Flags.GetValueOrDefault("setup.squadrons-this-turn") >= 2 ||
                !game.Topology.Squadrons.Any(x => x.Owner == game.Flow.CurrentPlayer && !Placed(game, x.EntityIndex) && !game.IsReserved(x.EntityIndex)))
            { game.Flow.CurrentPlayer = 3 - game.Flow.CurrentPlayer; game.Flow.Flags["setup.squadrons-this-turn"] = 0; }
            return;
        }
        var pending = Current(game) ?? throw new InvalidOperationException("Missing setup continuation.");
        var card = game.State.Upgrades[pending.Target];
        if (action.Kind == GameActionKind.Pass) { card.MarkSetupInitialized(); RemoveCurrent(game); return; }
        if (pending.Rule is "setup.reserve" or "setup.scout")
        {
            string runtime = RuntimeId(game, action.Target);
            if (!card.StoreUnit(runtime)) throw new InvalidOperationException("Unit already assigned to setup card.");
            if (pending.Rule == "setup.reserve")
            {
                game.State.Registers[game.Topology.Registers.EntityReserved(action.Target)] = 1;
                game.State.Registers[game.Topology.Registers[$"entity.{action.Target}.reserveUpgradePlusOne"]] = pending.Target + 1;
                game.State.Registers[game.Topology.Registers[$"entity.{action.Target}.reserveSourcePlusOne"]] = pending.Source + 1;
            }
            else game.State.Registers[game.Topology.Registers[$"entity.{action.Target}.scoutGranted"]] = 1;
            UpdateCurrent(game, pending with { Stage = pending.Stage + 1 });
            if (pending.Stage + 1 >= pending.Value) { card.MarkSetupInitialized(); RemoveCurrent(game); }
            return;
        }
        if (pending.Rule == "setup.place-token")
        {
            PlaceSetupToken(game, pending, action);
            if (pending.Text == "proximity-mines" && !card.Discarded) card.Discard();
            UpdateCurrent(game, pending with { Stage = pending.Stage + 1 });
            if (pending.Stage + 1 >= pending.Value) { card.MarkSetupInitialized(); RemoveCurrent(game); }
            return;
        }
        if (pending.Rule == "setup.redeploy")
        { PlaceEntity(game, pending.Source, action, redeploy: true); RemoveCurrent(game); return; }
        if (pending.Rule == "setup.move-obstacle")
        {
            if (!MapMovementRules.Enabled(game)) throw new ArgumentException("Obstacle relocation is disabled for this match.", nameof(action));
            int index = pending.Indices![0];
            var pose = new Pose(action.Position!.Value, action.YawDegrees);
            OfficialObjectives.MoveObstacle(game, index, pose, TabletopMeasurements.DistanceBand(2), flipped: action.Flipped, allowSquadronOverlap: true);
            RemoveCurrent(game);
            OfficialCreatureObjectives.BeginFlatObstacleDisplacements(game, index, game.State.Owners[pending.Source], GamePhase.Deployment); return;
        }
        var state = game.IsShip(pending.Source) ? game.State.Ships[pending.Source] : null;
        if (pending.Rule == "setup.schedule")
        {
            var selected = (pending.Indices ?? Array.Empty<int>()).Append(action.Amount).ToArray();
            if (selected.Length == 2) { card.ScheduleRounds(selected); RemoveCurrent(game); }
            else UpdateCurrent(game, pending with { Indices = selected });
            return;
        }
        if (pending.Rule == "setup.store-commands")
        {
            card.AddTypedCommandToken(action.Command);
            var selected = (pending.Indices ?? Array.Empty<int>()).Append((int)action.Command).ToArray();
            if (selected.Length >= pending.Value) { card.MarkSetupInitialized(); RemoveCurrent(game); }
            else UpdateCurrent(game, pending with { Indices = selected });
            return;
        }
        if (pending.Rule == "setup.assign-command") card.SetAssignedCommandToken(action.Command);
        else if (pending.Rule == "setup.replace-token") state!.TryReplaceDefenseToken(action.Token, ReplacementToken(pending.Text));
        else if (pending.Rule == "setup.gain-token") state!.AddDefenseToken(action.Token);
        else if (pending.Rule == "setup.varnillian") { card.StoreDie(AttackDieColor.Red, (DieFace)action.Index); card.MarkSetupInitialized(); }
        else if (pending.Rule == "setup.enemy-speed") game.State.Ships[action.Target].SetSpeed(action.Speed, game.Topology.Ships[action.Target].CatalogEntry.Definition.MaximumSpeed);
        else if (pending.Rule == "setup.mercy")
        {
            card.StoreUnit(RuntimeId(game, action.Target));
            game.State.Registers[game.Topology.Registers.UpgradeOwner(pending.Target)] = action.Target + 1;
        }
        else if (pending.Rule == "setup.aspiration")
        {
            if (action.AbilityId == "setup.aspiration.destination")
            { UpdateCurrent(game, pending with { Stage = (int)action.ToZone + 1, Value = 0 }); return; }
            if (action.AbilityId == "setup.aspiration.finish-zone")
            {
                var destinations = pending.Indices ?? Array.Empty<int>();
                if (pending.Value > 0) destinations = destinations.Append(pending.Stage - 1).ToArray();
                UpdateCurrent(game, pending with { Stage = 0, Value = 0, Indices = destinations }); return;
            }
            if (state!.TransferShield(action.FromZone, action.ToZone, maximumOverride: 6) < 1) throw new ArgumentException("No setup shield can transfer.");
            UpdateCurrent(game, pending with { Value = pending.Value + 1 }); return;
        }
        else if (pending.Rule == "setup.ozzel")
        {
            card.MarkSetupInitialized(); RemoveCurrent(game);
            var forced = action with { AbilityId = "movement.forced", Index = action.Amount, Amount = 0 };
            MovementModule.BeginForcedManeuver(game, pending.Source, forced, true, game.Flow.Phase); return;
        }
        else throw new NotSupportedException("Unhandled setup continuation: " + pending.Rule);
        card.MarkSetupInitialized(); RemoveCurrent(game);
    }

    private static void EnsureInitialized(GameContext game)
    {
        if (game.Flow.Flags.ContainsKey("setup.engine-stage")) return;
        game.Flow.Flags["setup.engine-stage"] = 0; game.Flow.CurrentPlayer = game.Topology.FirstPlayer;
        foreach (var scenario in game.Topology.Ships.Where(x => x.IsScenarioUnit)) SetPlaced(game, scenario.EntityIndex);
        foreach (var upgrade in game.Topology.Upgrades.Where(x => !x.IsSquadronLeader)
            .OrderBy(x => x.DefinitionId == "mercy-mission" ? 0 : 1)
            .ThenBy(x => x.Owner == game.Topology.FirstPlayer ? 0 : 1))
        {
            // A Grav Shift marker only serves the later obstacle relocation.
            // Do not schedule a useless placement decision in fixed-map matches.
            if (upgrade.DefinitionId == "grav-shift-reroute" && !MapMovementRules.Enabled(game)) continue;
            int source = upgrade.InitialEntityIndex;
            int squadronValue = CommandUpgradeEffects.SquadronValue(game.Topology.Ships[source].CatalogEntry.Definition.Squadron,
                game.HasUpgrade(source, "expanded-hangar-bay"), game.HasUpgrade(source, "spha-t"));
            var rule = upgrade.DefinitionId switch
            {
                "admiral-raddus-commander" or "profundity-title" or "rapid-launch-bays" => "setup.reserve",
                "hyperspace-rings" => "setup.scout",
                "g7-x-grav-well-projector" or "grav-shift-reroute" or "proximity-mines" => "setup.place-token",
                "mercy-mission" => "setup.mercy",
                _ => null
            };
            int maximum = upgrade.DefinitionId switch
            { "rapid-launch-bays" or "hyperspace-rings" => squadronValue, "proximity-mines" => game.Topology.Ships[source].CatalogEntry.Definition.Engineering / 2, _ => 1 };
            if (rule is not null && maximum > 0) game.Flow.Continuations.Add(new(rule, source, upgrade.UpgradeIndex, Value: maximum, Text: upgrade.DefinitionId));
        }
    }

    private static IReadOnlyList<GameActionOption> ChoiceOptions(GameContext game, GameContinuation pending)
    {
        var options = new List<GameActionOption>(); int source = pending.Source, upgrade = pending.Target;
        int owner = game.State.Owners[source]; var card = game.State.Upgrades[upgrade];
        if (!MapMovementRules.Enabled(game) && (pending.Rule == "setup.move-obstacle"
            || pending.Rule == "setup.place-token" && pending.Text == "grav-shift-reroute")) return options;
        if (pending.Rule is "setup.reserve" or "setup.scout")
        {
            for (int entity = 0; entity < game.Topology.EntityCount; entity++)
            {
                if (entity == source || game.State.Owners[entity] != owner || game.IsReserved(entity) || Placed(game, entity) || card.StoredUnitIds.Contains(RuntimeId(game, entity))) continue;
                bool eligible = pending.Text switch
                {
                    "admiral-raddus-commander" => game.IsShip(entity) && !game.Topology.Ships[entity].IsScenarioUnit,
                    "profundity-title" => game.IsShip(entity) && game.Topology.Ships[entity].CatalogEntry.Metadata.Size == "small" && game.Topology.Ships[entity].CatalogEntry.Definition.Command == 1,
                    "rapid-launch-bays" => !game.IsShip(entity),
                    "hyperspace-rings" => !game.IsShip(entity) && CommandModule.KeywordValue(game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition.Keywords, "ROGUE") == 0,
                    _ => false
                };
                if (eligible) Add(options, new(GameActionKind.ChooseEntity, source, entity, Index: upgrade, AbilityId: pending.Rule), "Choose " + RuntimeId(game, entity));
            }
            Pass(options, source, upgrade, "Finish " + card.Definition.name); return options;
        }
        if (pending.Rule == "setup.place-token")
        { options.Add(new(new(GameActionKind.PlaceToken, source, Index: upgrade, AbilityId: pending.Rule), "Place " + card.Definition.name + " token", AcceptsPosition: true)); Pass(options, source, upgrade, "Skip remaining setup tokens"); return options; }
        if (pending.Rule == "setup.redeploy")
        { options.Add(new(new(GameActionKind.Deploy, source, Index: upgrade, AbilityId: pending.Rule), "Redeploy Corvus", AcceptsPosition: true, AcceptsYaw: true)); Pass(options, source, upgrade, "Keep current position"); return options; }
        if (pending.Rule == "setup.move-obstacle")
        { options.Add(new(new(GameActionKind.PlaceToken, source, Index: upgrade, AbilityId: pending.Rule, Flipped: game.State.SpatialObjects[pending.Indices![0]].Flipped), "Move eligible obstacle", AcceptsPosition: true, AcceptsYaw: true, AcceptsFlip: true)); Pass(options, source, upgrade, "Leave obstacle"); return options; }
        if (pending.Rule == "setup.schedule")
        {
            for (int round = 1; round <= 6; round++)
                if (!(pending.Indices ?? Array.Empty<int>()).Any(x => Math.Abs(x - round) <= 1))
                    Add(options, new(GameActionKind.UseAbility, source, Index: upgrade, Amount: round, AbilityId: pending.Rule), "Schedule round " + round);
        }
        else if (pending.Rule is "setup.store-commands" or "setup.assign-command")
        {
            foreach (ShipCommand command in Enum.GetValues<ShipCommand>())
            {
                var selected = pending.Indices ?? Array.Empty<int>();
                if (pending.Rule == "setup.store-commands" && (pending.Text is "count-dooku-commander" or "resolute-title") &&
                    selected.Length == pending.Value - 1 && selected.Distinct().Count() == 1 && selected[0] == (int)command) continue;
                Add(options, new(GameActionKind.ChooseCommand, source, Command: command, Index: upgrade, AbilityId: pending.Rule), "Choose " + command);
            }
        }
        else if (pending.Rule is "setup.replace-token" or "setup.gain-token")
        {
            foreach (DefenseTokenType token in Enum.GetValues<DefenseTokenType>())
            {
                if (pending.Rule == "setup.gain-token" ? token == DefenseTokenType.Scatter : token == ReplacementToken(pending.Text) || !game.State.Ships[source].HasDefenseToken(token)) continue;
                Add(options, new(GameActionKind.ChooseToken, source, Token: token, Index: upgrade, AbilityId: pending.Rule), "Choose " + token);
            }
        }
        else if (pending.Rule == "setup.varnillian")
        {
            foreach (DieFace face in Enum.GetValues<DieFace>().Where(x => x != DieFace.HitCritical))
                Add(options, new(GameActionKind.ChooseDie, source, Index: (int)face, AbilityId: pending.Rule), "Store red " + face);
        }
        else if (pending.Rule == "setup.enemy-speed")
        {
            foreach (var target in game.Topology.Ships.Where(x => x.Owner != owner && !game.IsReserved(x.EntityIndex)))
                foreach (int change in new[] { -1, 1 })
                {
                    int speed = game.State.Ships[target.ShipIndex].CurrentSpeed + change;
                    if (speed >= 0 && speed <= target.CatalogEntry.Definition.MaximumSpeed)
                        Add(options, new(GameActionKind.ChangeSpeed, source, target.EntityIndex, Index: upgrade, Speed: speed, AbilityId: pending.Rule), "Set enemy speed " + speed);
                }
            Pass(options, source, upgrade, "Skip Admiral Titus");
        }
        else if (pending.Rule == "setup.mercy")
        {
            foreach (var target in game.Topology.Ships.Where(x => x.Owner == owner && !game.IsReserved(x.EntityIndex) &&
                x.CatalogEntry.Metadata.Size == "small" && !game.UpgradesFor(x.EntityIndex).Any(c => c.Definition.slot == "commander")))
                Add(options, new(GameActionKind.ChooseEntity, source, target.EntityIndex, Index: upgrade, AbilityId: pending.Rule), "Assign Mercy Mission");
        }
        else if (pending.Rule == "setup.aspiration") AspirationOptions(game, pending, options);
        else if (pending.Rule == "setup.ozzel")
        {
            foreach (var course in MovementModule.GetForcedManeuverActions(game, source, true))
                options.Add(course with { Template = course.Template with { AbilityId = pending.Rule, Index = upgrade, Amount = course.Template.Index, Speed = 1 } });
            Pass(options, source, upgrade, "Decline Ozzel maneuver");
        }
        else throw new NotSupportedException("Unsupported setup choice: " + pending.Rule);
        return options;
    }

    private static void PreparePostDeployment(GameContext game)
    {
        game.Flow.Flags["setup.engine-stage"] = 2;
        var scheduled = new List<GameContinuation>(); var choices = new List<GameContinuation>(); var firstRound = new List<GameContinuation>();
        foreach (var ship in game.Topology.Ships.Where(x => !game.IsReserved(x.EntityIndex) && !game.IsDestroyed(x.EntityIndex))
            .OrderBy(x => x.Owner == game.Topology.FirstPlayer ? 0 : 1))
        {
            int source = ship.EntityIndex;
            ApplyFixedSetupEffects(game, source);
            foreach (var upgrade in game.Topology.Upgrades.Where(x => x.InitialEntityIndex == source && !x.IsSquadronLeader))
            {
                string id = upgrade.DefinitionId; int index = upgrade.UpgradeIndex;
                if (id is "garm-bel-iblis-commander" or "general-tagge-commander" or "shu-mai-officer")
                { scheduled.Add(new("setup.schedule", source, index, Text: id)); continue; }
                string? rule = id switch
                {
                    "captain-needa-officer" or "vanguard-title" or "local-fire-control" => "setup.replace-token",
                    "kyrsta-agate-commander" => "setup.gain-token", "taskmaster-grint-officer" => "setup.assign-command",
                    "admiral-titus-officer" => "setup.enemy-speed",
                    "aspiration-title" => "setup.aspiration", "gunnery-chief-varnillian" => "setup.varnillian",
                    "admiral-tarkin-commander" or "admiral-trench-commander" or "count-dooku-commander" or "general-draven-commander" or
                        "grand-admiral-thrawn-commander" or "rune-haako-officer" or "resolute-title" => "setup.store-commands",
                    _ => null
                };
                if (id == "corvus-title") choices.Add(new("setup.redeploy", source, index, Text: id));
                if (id == "grav-shift-reroute" && MapMovementRules.Enabled(game))
                    foreach (var token in game.State.SpatialObjects.Where(x => x.Kind == GravShiftKind && x.SourceEntity == source))
                        for (int piece = 0; piece < game.State.SpatialObjects.Count; piece++)
                            if (game.State.SpatialObjects[piece] is { Active: true, Kind: 0 } && Geometry2D.Distance(new Circle(token.Pose.Position, MathF.Max(token.Size.X, token.Size.Y) / 2), ObstacleGeometry.Shape(game, piece)) <= TabletopMeasurements.MediumRange)
                                choices.Add(new("setup.move-obstacle", source, index, Indices: new[] { piece }));
                if (id == "admiral-ozzel-officer") firstRound.Add(new("setup.ozzel", source, index, Text: id));
                if (rule is null) continue;
                var card = game.State.Upgrades[index];
                if (rule == "setup.varnillian" && ship.Owner == game.Topology.FirstPlayer)
                { card.StoreDie(AttackDieColor.Red, DieFace.Blank); card.MarkSetupInitialized(); continue; }
                int required = id switch { "admiral-tarkin-commander" => 6, "resolute-title" => 4, _ => Math.Max(1, card.Definition.startCommandAmount) };
                if (rule == "setup.store-commands") card.ClearTypedCommandTokens();
                choices.Add(new(rule, source, index, Value: rule == "setup.store-commands" ? required : 0, Text: id));
            }
        }
        game.Flow.Continuations.AddRange(scheduled); game.Flow.Continuations.AddRange(choices);
        game.Flow.Continuations.AddRange(firstRound);
    }

    public static void ApplyFixedSetupEffects(GameContext game, int ship)
    {
        if (game.Flow.Flags.GetValueOrDefault($"setup.fixed.{ship}") != 0) return;
        game.Flow.Flags[$"setup.fixed.{ship}"] = 1;
        if (game.Topology.Ships[ship].IsStation) return;
        using var scope = new GameScope(game);
        UnitAbilityEngine.ApplyFixedSetupEffects(game.ShipViews[ship]);
        game.State.Ships[ship].IncreaseMaximumHull(CommanderRules.HullBonus(game.ShipViews[ship]));
    }

    private static IReadOnlyList<GameActionOption> DeploymentOptions(GameContext game, int owner)
    {
        var result = new List<GameActionOption>(); bool squadOnly = game.Flow.Flags.GetValueOrDefault("setup.squadrons-this-turn") == 1;
        for (int entity = 0; entity < game.Topology.EntityCount; entity++)
            if (game.State.Owners[entity] == owner && !Placed(game, entity) && !game.IsReserved(entity) && (!squadOnly || !game.IsShip(entity)))
            {
                if (!OfficialObjectives.CanOfferDeployment(game, entity)) continue;
                result.Add(new(new(GameActionKind.Deploy, entity, Speed: game.IsShip(entity) ? game.State.Ships[entity].CurrentSpeed : 0),
                    "Deploy " + RuntimeId(game, entity), AcceptsPosition: true, AcceptsYaw: game.IsShip(entity)));
                if (!game.IsShip(entity) && CommandModule.KeywordValue(game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition.Keywords, "SWARM") > 0)
                    foreach (var source in game.Topology.Ships.Where(x => x.Owner == owner && Placed(game, x.EntityIndex) && !game.IsReserved(x.EntityIndex)))
                        foreach (var upgrade in game.Topology.Upgrades.Where(x => x.InitialEntityIndex == source.EntityIndex && x.DefinitionId == "invisible-hand-title" &&
                            game.State.Upgrades[x.UpgradeIndex].Active && game.State.Upgrades[x.UpgradeIndex].StoredUnitIds.Count < 5))
                            Add(result, new(GameActionKind.ChooseEntity, source.EntityIndex, entity, Index: upgrade.UpgradeIndex, AbilityId: "setup.invisible-hand.reserve"), "Set aside Swarm squadron next to Invisible Hand");
            }
        return result;
    }

    private static void PlaceEntity(GameContext game, int entity, GameAction action, bool redeploy)
    {
        if (action.Position is null) throw new ArgumentException("A deployment position is required.");
        var pose = new Pose(action.Position.Value, action.YawDegrees);
        if (game.IsShip(entity))
        {
            if (!LegalShipPose(game, entity, pose, true)) throw new ArgumentException("Illegal ship deployment pose.");
            int maximum = game.Topology.Ships[entity].CatalogEntry.Definition.ChartedMaximumSpeed;
            if (action.Speed < 0 || action.Speed > maximum) throw new ArgumentException("Illegal deployment speed.");
            OfficialObjectives.ValidateDeploymentSpeed(game, entity, action.Speed);
            int speed = action.Speed;
            var shape = new OrientedRectangle(pose, game.Topology.Ships[entity].FootprintMillimetres);
            if (game.State.SpatialObjects.Any(x => x.Active && x.Kind == GravWellKind && Vector2.Distance(x.Pose.Position, shape.ClosestPoint(x.Pose.Position)) <= TabletopMeasurements.MediumRange)) speed = 0;
            if (OfficialObjectives.Obstacles(game).Any(i => OfficialObjectives.ObstacleId(game, i) == "gravity-rift" &&
                Geometry2D.Distance(shape, ObstacleGeometry.Shape(game, i)) <= TabletopMeasurements.DistanceBand(2))) speed = 0;
            game.State.Ships[entity].SetSpeed(speed, maximum);
        }
        else if (!LegalSquadronPose(game, entity, pose)) throw new ArgumentException("Illegal squadron deployment pose.");
        game.State.Poses[entity] = pose; SetPlaced(game, entity);
        if (!redeploy && game.IsShip(entity)) OfficialObjectives.RecordShipDeployment(game, entity);
    }

    public static bool LegalShipPose(GameContext game, int entity, Pose pose, bool requireDeploymentZone)
    {
        var shape = new OrientedRectangle(pose, game.Topology.Ships[entity].FootprintMillimetres);
        if (!InsideBoard(game, shape)) return false;
        if (requireDeploymentZone)
        {
            var zone = OfficialObjectives.State(game) is null ? Zone(game, game.State.Owners[entity]) : OfficialObjectives.DeploymentZone(game, game.State.Owners[entity]);
            if (OfficialObjectives.State(game) is not null && !Contains(OfficialObjectives.SetupArea(game), shape)) return false;
            bool? objectivePlacement = OfficialObjectives.CheckShipDeployment(game, entity, shape);
            if (objectivePlacement == false) return false;
            if (game.Topology.Ships[entity].CatalogEntry.Definition.HasAuxiliaryHullZones &&
                !(objectivePlacement == true && (OfficialObjectives.Is(game, "fleet-ambush") ||
                    OfficialObjectives.Is(game, "surprise-attack") && OfficialObjectives.IsFlagship(game, entity))))
            {
                Span<Vector2> corners = stackalloc Vector2[4]; shape.CopyCorners(corners);
                var own = game.Topology.DeploymentZones.Single(x => x.Owner == game.State.Owners[entity]);
                var enemy = game.Topology.DeploymentZones.Single(x => x.Owner != own.Owner);
                var outward = (own.MinimumMillimetres + own.MaximumMillimetres - enemy.MinimumMillimetres - enemy.MaximumMillimetres) * .5f;
                float edgeDistance = OfficialObjectives.State(game) is not null ? OfficialObjectives.DistanceFromPlayerEdge(game, shape, game.State.Owners[entity])
                    : MathF.Abs(outward.X) > MathF.Abs(outward.Y)
                    ? MathF.Abs(corners.ToArray().Max(x => x.X * MathF.Sign(outward.X)) - game.Topology.BoardSizeMillimetres.X * .5f)
                    : MathF.Abs(corners.ToArray().Max(x => x.Y * MathF.Sign(outward.Y)) - game.Topology.BoardSizeMillimetres.Y * .5f);
                if (edgeDistance > .2285f) return false;
            }
            else if (objectivePlacement != true && !Contains(zone, shape)) return false;
        }
        return ClearOfPlacedPieces(game, entity, shape);
    }

    private static bool LegalSquadronPose(GameContext game, int entity, Pose pose)
    {
        var source = game.Topology.Squadrons[game.SquadronIndex(entity)];
        var circle = new Circle(pose.Position, source.FootprintMillimetres.X * .5f);
        var half = game.Topology.BoardSizeMillimetres * .5f;
        if (MathF.Abs(circle.Center.X) + circle.Radius > half.X || MathF.Abs(circle.Center.Y) + circle.Radius > half.Y) return false;
        if (OfficialObjectives.State(game) is not null && !PlayAreaGeometry.Contains(OfficialObjectives.SetupArea(game), circle)) return false;
        bool scout = CommandModule.KeywordValue(source.CatalogEntry.Definition.Keywords, "SCOUT") > 0 || game.State.Registers[game.Topology.Registers[$"entity.{entity}.scoutGranted"]] != 0;
        if (scout)
        {
            for (int other = 0; other < game.Topology.EntityCount; other++)
                if (Placed(game, other) && game.State.Owners[other] != source.Owner && !game.IsReserved(other) &&
                    (game.IsShip(other) ? Geometry2D.Distance(circle, game.ShipBase(other)) : Geometry2D.Distance(circle, game.SquadronBase(other))) <= TabletopMeasurements.LongRange) return false;
        }
        else if (!game.Topology.Ships.Any(ship => ship.Owner == source.Owner && !ship.IsScenarioUnit && Placed(game, ship.EntityIndex) && !game.IsReserved(ship.EntityIndex) &&
            Geometry2D.Distance(circle, game.ShipBase(ship.EntityIndex)) <= TabletopMeasurements.DistanceBand(2) &&
            (!ship.CatalogEntry.Definition.HasAuxiliaryHullZones || Contains(Zone(game, source.Owner), game.ShipBase(ship.EntityIndex)) ||
                Geometry2D.Distance(circle, Zone(game, source.Owner)) <= TabletopMeasurements.DistanceBand(2)))) return false;
        for (int other = 0; other < game.Topology.EntityCount; other++)
            if (other != entity && Placed(game, other) && !game.IsReserved(other) &&
                (game.IsShip(other) ? Geometry2D.Overlaps(circle, game.ShipBase(other)) : Geometry2D.Overlaps(circle, game.SquadronBase(other)))) return false;
        // Ordinary squadron deployment may overlap obstacles without resolving them.
        return !OfficialObjectives.Is(game, "fleet-ambush") || source.Owner != OfficialObjectives.FirstPlayer(game) || !OfficialObjectives.Obstacles(game).Any(index =>
            Geometry2D.Overlaps(circle, ObstacleGeometry.Shape(game, index)) && Geometry2D.Overlaps(OfficialObjectives.AmbushZone(game), ObstacleGeometry.Shape(game, index)));
    }

    private static bool ClearOfPlacedPieces(GameContext game, int source, OrientedRectangle shape)
    {
        for (int entity = 0; entity < game.Topology.EntityCount; entity++)
            if (entity != source && Placed(game, entity) && !game.IsReserved(entity) && EntityOverlaps(game, entity, shape)) return false;
        return !Enumerable.Range(0, game.State.SpatialObjects.Count).Any(index =>
            game.State.SpatialObjects[index] is { Active: true, Kind: 0 } && Geometry2D.Overlaps(shape, ObstacleGeometry.Shape(game, index)) &&
            !(OfficialObjectives.Is(game, "surprise-attack") && game.State.Owners[source] == OfficialObjectives.FirstPlayer(game) &&
                OfficialObjectives.IsFlagship(game, source) && OfficialObjectives.ObstacleId(game, index) == "station"));
    }
    private static bool EntityOverlaps(GameContext game, int entity, OrientedRectangle shape) => game.IsShip(entity) ? Geometry2D.Overlaps(shape, game.ShipBase(entity)) : Geometry2D.Overlaps(game.SquadronBase(entity), shape);
    private static bool InsideBoard(GameContext game, OrientedRectangle shape) => Contains(new OrientedRectangle(new Pose(Vector2.Zero), game.Topology.BoardSizeMillimetres), shape);
    private static bool Contains(OrientedRectangle outer, OrientedRectangle inner) { Span<Vector2> corners = stackalloc Vector2[4]; inner.CopyCorners(corners); foreach (var corner in corners) if (!outer.Contains(corner, .0115f)) return false; return true; }
    private static OrientedRectangle Zone(GameContext game, int owner) { var zone = game.Topology.DeploymentZones.Single(x => x.Owner == owner); return new(new Pose((zone.MinimumMillimetres + zone.MaximumMillimetres) * .5f), zone.MaximumMillimetres - zone.MinimumMillimetres); }

    private static void PlaceSetupToken(GameContext game, GameContinuation pending, GameAction action)
    {
        if (pending.Text == "grav-shift-reroute" && !MapMovementRules.Enabled(game))
            throw new ArgumentException("Grav Shift setup is disabled for this match.", nameof(action));
        if (action.Position is null) throw new ArgumentException("Setup token position is required.");
        var position = action.Position.Value; Geometry2D.RequireFinite(position, nameof(action));
        int kind = pending.Text switch { "g7-x-grav-well-projector" => GravWellKind, "grav-shift-reroute" => GravShiftKind, _ => ProximityMineKind };
        float radius = kind == ProximityMineKind ? TabletopMeasurements.FromWorldUnits(.26f) : .5f;
        var half = game.Topology.BoardSizeMillimetres * .5f;
        if (MathF.Abs(position.X) + radius > half.X || MathF.Abs(position.Y) + radius > half.Y) throw new ArgumentException("Setup token must remain on the board.");
        if (kind == ProximityMineKind && game.State.SpatialObjects.Any(x => x.Active && x.Kind == kind && Vector2.Distance(position, x.Pose.Position) - radius * 2 <= TabletopMeasurements.DistanceOne))
            throw new ArgumentException("Proximity mines must be beyond distance one of one another.");
        game.State.SpatialObjects.Add(new(kind, new Pose(position), new Vector2(radius * 2), game.State.Owners[pending.Source], pending.Source, Payload: pending.Target + 1));
        TokenObstacleAttachments.RecordPlacement(game, game.State.SpatialObjects.Count - 1);
    }

    private static void AspirationOptions(GameContext game, GameContinuation pending, List<GameActionOption> options)
    {
        var state = game.State.Ships[pending.Source];
        if (pending.Stage > 0)
        {
            var destination = (HullZone)(pending.Stage - 1);
            if (state.GetShields(destination) < 6)
                foreach (var source in state.HullZones.Where(x => x != destination && state.GetShields(x) > 0))
                    Add(options, new(GameActionKind.TransferShield, pending.Source, FromZone: source, ToZone: destination, Index: pending.Target, AbilityId: "setup.aspiration.transfer"), "Move one shield to " + destination);
            Add(options, new(GameActionKind.UseAbility, pending.Source, Index: pending.Target, AbilityId: "setup.aspiration.finish-zone"), "Finish destination zone");
        }
        else
        {
            if ((pending.Indices?.Length ?? 0) < 2)
                foreach (var destination in state.HullZones.Where(x => !(pending.Indices ?? Array.Empty<int>()).Contains((int)x) && state.GetShields(x) < 6 && state.HullZones.Any(s => s != x && state.GetShields(s) > 0)))
                    Add(options, new(GameActionKind.ChooseZone, pending.Source, ToZone: destination, Index: pending.Target, AbilityId: "setup.aspiration.destination"), "Choose " + destination + " destination");
            Pass(options, pending.Source, pending.Target, "Finish Aspiration");
        }
    }

    private static void FinishSetup(GameContext game)
    {
        if (OfficialObjectives.Is(game, "fleet-ambush"))
            foreach (int token in OfficialObjectives.TokenIndices(game).Where(i => game.State.SpatialObjects[i].Id == "fleet-ambush-corner").ToArray())
                OfficialObjectives.RemoveToken(game, token);
        game.Flow.Flags["setup.engine-stage"] = 3;
        game.State.Registers[game.Topology.Registers["setup.completed"]] = 1;
        if (game.State.Objective is EscortVipObjective vip) vip.BeginMatch();
        else if (game.State.Objective is IntelRetrievalObjective intel) intel.BeginMatch();
        else if (game.State.Objective is SeizeStationObjective station) station.BeginMatch();
        game.Flow.Phase = GamePhase.StartRound; game.Flow.PlanningShip = 0;
        game.Flow.CurrentPlayer = game.Topology.FirstPlayer;
        ReactionAbilityEffects.SnapshotSpeeds(game);
    }
    private static DefenseTokenType ReplacementToken(string id) => id switch { "captain-needa-officer" => DefenseTokenType.Evade, "vanguard-title" => DefenseTokenType.Redirect, _ => DefenseTokenType.Salvo };
    private static bool Placed(GameContext game, int entity) => game.State.Registers[game.Topology.Registers[$"entity.{entity}.placed"]] != 0;
    private static void SetPlaced(GameContext game, int entity) => game.State.Registers[game.Topology.Registers[$"entity.{entity}.placed"]] = 1;
    private static string RuntimeId(GameContext game, int entity) => game.IsShip(entity) ? game.Topology.Ships[entity].RuntimeId : game.Topology.Squadrons[game.SquadronIndex(entity)].RuntimeId;
    private static int CurrentIndex(GameContext game)
    {
        int selected = (int)game.Flow.Flags.GetValueOrDefault("setup.timing.selected") - 1;
        return selected >= 0 && selected < game.Flow.Continuations.Count && game.Flow.Continuations[selected].Rule.StartsWith("setup.", StringComparison.Ordinal)
            ? selected : game.Flow.Continuations.FindIndex(x => x.Rule.StartsWith("setup.", StringComparison.Ordinal));
    }
    private static GameContinuation? Current(GameContext game) => CurrentIndex(game) is var index && index >= 0 ? game.Flow.Continuations[index] : null;
    private static void RemoveCurrent(GameContext game)
    {
        int index = CurrentIndex(game); if (index >= 0) game.Flow.Continuations.RemoveAt(index);
        bool ordered = game.Flow.Flags.Remove("setup.timing.selected");
        if (ordered && game.Flow.Continuations.LastOrDefault() is { Rule: "effects.window" } window && window.Text.StartsWith("setup-", StringComparison.Ordinal))
            game.Flow.Phase = GamePhase.AbilityChoice;
    }
    private static void UpdateCurrent(GameContext game, GameContinuation value) => game.Flow.Continuations[CurrentIndex(game)] = value;
    private static void Add(List<GameActionOption> options, GameAction action, string label) => options.Add(new(action, label));
    private static void Pass(List<GameActionOption> options, int source, int upgrade, string label) => Add(options, new(GameActionKind.Pass, source, Index: upgrade, AbilityId: "setup.finish-choice"), label);
    private static bool Matches(GameActionOption option, GameAction action)
    {
        var template = option.Template;
        var normalized = action with { Position = option.AcceptsPosition ? null : action.Position, YawDegrees = option.AcceptsYaw ? template.YawDegrees : action.YawDegrees,
            Speed = option.AcceptsPosition && action.Kind == GameActionKind.Deploy ? template.Speed : action.Speed,
            Course = template.Course is null ? action.Course : template.Course };
        return template == normalized && (template.Course is null || action.Course is not null && template.Course.SequenceEqual(action.Course));
    }
}
