using System.Collections.ObjectModel;
using System.Numerics;
using System.Security.Cryptography;
using System.Text.Json;
using Armada.Core;
using Armada.Data;
using Armada.Fleet;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless.Catalog;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Setup;

public enum SetupRequirementStage { Deployment, PostDeployment, BeforeFirstRound }
public sealed record PendingSetupRequirement(SetupRequirementStage Stage, string Rule,
    int Owner = 0, int SourceEntity = -1, int UpgradeIndex = -1);

public sealed record PreparedShipTopology(int EntityIndex, int ShipIndex, int Owner,
    string RuntimeId, string FleetEntryId, ShipCatalogEntry CatalogEntry,
    IReadOnlyList<int> UpgradeIndices, Vector2 FootprintMillimetres, bool IsScenarioUnit = false, bool IsStation = false);
public sealed record PreparedSquadronTopology(int EntityIndex, int SquadronIndex, int Owner,
    string RuntimeId, string FleetEntryId, SquadronCatalogEntry CatalogEntry,
    int LeaderUpgradeIndex, Vector2 FootprintMillimetres);
public sealed record PreparedUpgradeTopology(int UpgradeIndex, int InitialEntityIndex,
    int Owner, string DefinitionId, bool IsSquadronLeader);
public sealed record PreparedMapPlacement(int SpatialIndex, string PlacementId, string ContentId,
    string DisplayName, MapPlacementKind Kind, float InteractionRangeMillimetres, int StartingSpeed, bool UseCustomSize = false);
public sealed record PreparedDeploymentZone(int Owner, Vector2 MinimumMillimetres, Vector2 MaximumMillimetres);

/// <summary>Named indices make every initialized setup flag and future card transfer snapshot-owned.</summary>
public sealed class MatchRegisterLayout
{
    private readonly Dictionary<string, int> indices;
    private readonly int[] reservedIndices;
    private readonly int[] upgradeOwnerIndices;
    public IReadOnlyList<string> Names { get; }
    public int Count => Names.Count;
    internal MatchRegisterLayout(int entities, int upgrades)
    {
        var names = new List<string>
        {
            "setup.stage", "setup.completed", "setup.currentPlayer", "setup.choiceActive",
            "setup.choiceSourcePlusOne", "setup.choiceUpgradePlusOne", "setup.choiceSelectedCount",
            "setup.scheduledRoundChoiceActive", "setup.scheduledRoundOne", "setup.scheduledRoundTwo",
            "setup.postDeploymentApplied", "setup.objectiveSelectionCompleted",
            "round.started", "round.commandPlanningCompleted", "round.fleetCommandMask.player1",
            "round.fleetCommandMask.player2", "round.victoryTokens.player1", "round.victoryTokens.player2"
        };
        for (int entity = 0; entity < entities; entity++)
            foreach (string field in new[] { "placed", "reserved", "reserveUpgradePlusOne", "reserveSourcePlusOne", "scoutGranted" })
                names.Add($"entity.{entity}.{field}");
        for (int upgrade = 0; upgrade < upgrades; upgrade++) names.Add($"upgrade.{upgrade}.ownerEntityPlusOne");
        Names = names.AsReadOnly();
        indices = names.Select((name, index) => (name, index)).ToDictionary(x => x.name, x => x.index, StringComparer.Ordinal);
        reservedIndices = Enumerable.Range(0, entities).Select(entity => indices[$"entity.{entity}.reserved"]).ToArray();
        upgradeOwnerIndices = Enumerable.Range(0, upgrades).Select(upgrade => indices[$"upgrade.{upgrade}.ownerEntityPlusOne"]).ToArray();
    }
    public int this[string name] => indices.TryGetValue(name, out int index) ? index :
        throw new KeyNotFoundException("Unknown match register: " + name);
    public int UpgradeOwner(int upgradeIndex) => upgradeOwnerIndices[upgradeIndex];
    public int EntityReserved(int entityIndex) => reservedIndices[entityIndex];
}

public sealed class MatchTopology
{
    public string MatchId { get; }
    public string Ruleset { get; }
    public int FirstPlayer { get; }
    public int EscortPlayer { get; }
    public string ScenarioId { get; }
    public string ObjectiveId { get; }
    public bool MapMovementEnabled { get; }
    public IReadOnlyList<string> RemovedFromPlayAreaKeys { get; }
    public IReadOnlyList<string> GravityRestoreKeys { get; }
    public IReadOnlyList<string> ProjectorRestoreKeys { get; }
    public bool HasMoraloEval { get; }
    public Vector2 BoardSizeMillimetres { get; }
    public float PlayPlaneMillimetres { get; }
    public IReadOnlyList<PreparedShipTopology> Ships { get; }
    public IReadOnlyList<PreparedSquadronTopology> Squadrons { get; }
    public IReadOnlyList<PreparedUpgradeTopology> Upgrades { get; }
    public IReadOnlyList<PreparedMapPlacement> MapPlacements { get; }
    public IReadOnlyList<PreparedDeploymentZone> DeploymentZones { get; }
    public IReadOnlyDictionary<string, int> EntityByRuntimeId { get; }
    public MatchRegisterLayout Registers { get; }
    public int EntityCount => Ships.Count + Squadrons.Count;
    /// <summary>Reserves and revivals reuse existing indices. Any newly discovered entity-creation
    /// rule must allocate its full topology before snapshotting; implicit growth is unsupported.</summary>
    public bool AllowsImplicitEntityCreation => false;
    internal MatchTopology(MatchConfigDto config, PreparedShipTopology[] ships,
        PreparedSquadronTopology[] squadrons, PreparedUpgradeTopology[] upgrades,
        PreparedMapPlacement[] placements, MatchRegisterLayout registers)
    {
        MatchId = config.matchId; Ruleset = config.ruleset; FirstPlayer = config.firstPlayerId;
        EscortPlayer = config.escortPlayerId; ScenarioId = config.map.scenarioId;
        ObjectiveId = MapScenarioCatalog.Normalize(ScenarioId) == MapScenarioCatalog.DefaultId ? config.objectiveId : string.Empty;
        MapMovementEnabled = config.mapMovementEnabled;
        RemovedFromPlayAreaKeys = Array.AsReadOnly(Enumerable.Range(0, ships.Length + squadrons.Length).Select(i => "removed.play-area." + i).ToArray());
        GravityRestoreKeys = Array.AsReadOnly(Enumerable.Range(0, ships.Length).Select(i => "movement.gravity-restore." + i).ToArray());
        ProjectorRestoreKeys = Array.AsReadOnly(Enumerable.Range(0, ships.Length).Select(i => "movement.restore-speed." + i).ToArray());
        BoardSizeMillimetres = ConvertVector(config.map.widthWorldUnits, config.map.lengthWorldUnits);
        PlayPlaneMillimetres = TabletopMeasurements.FromWorldUnits(config.map.playPlaneY);
        Ships = Array.AsReadOnly(ships); Squadrons = Array.AsReadOnly(squadrons);
        HasMoraloEval = squadrons.Any(x => x.CatalogEntry.Definition.SquadronId == "moralo-eval-yv-666");
        Upgrades = Array.AsReadOnly(upgrades); MapPlacements = Array.AsReadOnly(placements);
        DeploymentZones = Array.AsReadOnly((config.map.deploymentZones ?? Array.Empty<DeploymentZoneDto>()).Select(x =>
            new PreparedDeploymentZone(x.playerId, ConvertVector(x.minimumX, x.minimumZ), ConvertVector(x.maximumX, x.maximumZ))).ToArray());
        var runtimeIds = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var ship in ships) runtimeIds.Add(ship.RuntimeId, ship.EntityIndex);
        foreach (var squadron in squadrons) runtimeIds.Add(squadron.RuntimeId, squadron.EntityIndex);
        EntityByRuntimeId = new ReadOnlyDictionary<string, int>(runtimeIds);
        Registers = registers;
    }
    internal static Vector2 ConvertVector(float x, float y) => new(TabletopMeasurements.FromWorldUnits(x), TabletopMeasurements.FromWorldUnits(y));
}

/// <summary>A fully allocated source baseline, pending the actual source setup decisions and rule port.</summary>
public sealed class PreparedMatch
{
    private readonly MatchConfigDto config;
    public BattleState InitialState { get; }
    public MatchTopology Topology { get; }
    public IReadOnlyList<PendingSetupRequirement> PendingSetupRequirements { get; }
    internal PreparedMatch(MatchConfigDto config, BattleState state, MatchTopology topology,
        PendingSetupRequirement[] pending)
    {
        this.config = MatchDocumentClone.Clone(config);
        InitialState = state; Topology = topology;
        PendingSetupRequirements = Array.AsReadOnly(pending);
    }
    public MatchConfigDto CaptureMatchConfig() => MatchDocumentClone.Clone(config);
    public BattleState CreateState() => InitialState.DeepClone();
}

public static partial class MatchPreparation
{
    public const int MatchSetupRuleId = 1000;
    public const string SchemaVersion = "swarmada-headless.prepared-match.v4";

    /// <summary>
    /// Initializes the source ShipView/SquadronView baseline, including card starting tokens and the
    /// exact two match-seeded damage decks. Unresolved deployment and post-deployment setup effects
    /// remain pending; preparation never guesses player decisions or advances to a playable round.
    /// </summary>
    public static PreparedMatch Prepare(MatchConfigDto match, FleetData data, ulong seed = 0)
    {
        ArgumentNullException.ThrowIfNull(match);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(match.map);
        if (!UpgradeCatalog.IsLoaded) throw new InvalidOperationException("Load UpgradeCatalog before preparing a match.");
        if (MapScenarioCatalog.Normalize(match.map.scenarioId) == MapScenarioCatalog.DefaultId &&
            !Engine.OfficialObjectives.SupportedIds.Contains(match.objectiveId, StringComparer.Ordinal))
            throw new ArgumentException("A default match requires exactly one supported objectiveId, supplied before setup.", nameof(match));
        var structural = SessionValidation.ValidateMatch(match, data);
        if (!structural.IsValid)
            throw new ArgumentException("Match failed source validation: " + string.Join(" ", structural.Errors), nameof(match));
        var fleetCatalog = data.BuildFleetCatalog();
        var ruleset = ResolveRuleset(match.ruleset);
        var validator = new FleetLegalityValidator(fleetCatalog, ruleset, new FleetValidationOptions(reportGameplayCoverage: false));
        foreach (var player in match.players)
        {
            var report = validator.Validate(player.fleet);
            if (!report.IsLegal)
                throw new ArgumentException($"Player {player.playerId} fleet is illegal: " +
                    string.Join(" ", report.Issues.Where(x => x.Severity == FleetLegalitySeverity.Error).Select(x => x.Code + ": " + x.Message)), nameof(match));
        }
        foreach (var piece in match.map.placements ?? Array.Empty<MapPlacementDto>())
            if (piece.kind is MapPlacementKind.Ship or MapPlacementKind.Squadron)
                throw new NotSupportedException("Sandbox unit placements must first be converted to fleet entries and UnitDeploymentDto records: " + piece.placementId);

        var config = MatchDocumentClone.Clone(match);
        foreach (var placement in config.map.placements ?? Array.Empty<MapPlacementDto>())
            if (placement.kind == MapPlacementKind.Obstacle)
            {
                var size = MapObstacleGeometry.Size(placement);
                placement.widthWorldUnits = size.X / TabletopMeasurements.MillimetresPerWorldUnit;
                placement.lengthWorldUnits = size.Y / TabletopMeasurements.MillimetresPerWorldUnit;
            }
        var players = config.players.ToDictionary(x => x.playerId);
        var shipsByDefinition = data.ShipEntries.ToDictionary(x => x.Definition.ShipId, StringComparer.OrdinalIgnoreCase);
        var squadronsByDefinition = data.SquadronEntries.ToDictionary(x => x.Definition.SquadronId, StringComparer.OrdinalIgnoreCase);
        foreach (var entry in data.SquadronEntries) squadronsByDefinition.TryAdd(entry.Id, entry);
        var shipDeployments = config.deployments.Where(x => x.unitKind == RuntimeUnitKind.Ship).ToArray();
        var squadronDeployments = config.deployments.Where(x => x.unitKind == RuntimeUnitKind.Squadron).ToArray();
        ShipCatalogEntry? vipEntry = null;
        var stations = new Dictionary<string, ShipCatalogEntry>(StringComparer.Ordinal);
        if (MapScenarioCatalog.Normalize(config.map.scenarioId) == MapScenarioCatalog.EscortVipId)
        {
            (vipEntry, var vipDeployment) = PrepareVip(config, data);
            shipDeployments = shipDeployments.Append(vipDeployment).ToArray();
        }
        if (MapScenarioCatalog.Normalize(config.map.scenarioId) == MapScenarioCatalog.DefaultId && config.objectiveId == "station-assault")
        {
            for (int ordinal = 0; ordinal < 2; ordinal++)
            {
                string entryId = "objective-unarmed-station-" + ordinal;
                stations.Add(entryId, CreateUnarmedStation(config));
                shipDeployments = shipDeployments.Append(new UnitDeploymentDto
                {
                    runtimeUnitId = RuntimeUnitId.ForMatchEntry(config.matchId, 3 - config.firstPlayerId, RuntimeUnitKind.Ship, entryId).Value,
                    fleetEntryInstanceId = entryId, ownerPlayerId = 3 - config.firstPlayerId, unitKind = RuntimeUnitKind.Ship
                }).ToArray();
            }
        }
        var shipStates = new List<ShipRuntimeState>();
        var squadronStates = new List<SquadronRuntimeState>();
        var commands = new List<CommandStackState>();
        var upgradeStates = new List<UpgradeCardRuntimeState>();
        var shipTopology = new List<PreparedShipTopology>();
        var squadronTopology = new List<PreparedSquadronTopology>();
        var upgradeTopology = new List<PreparedUpgradeTopology>();
        var pending = new List<PendingSetupRequirement>
        {
            new(SetupRequirementStage.Deployment, "deployment.source-sequence"),
            new(SetupRequirementStage.BeforeFirstRound, "commands.initial-planning")
        };
        if (MapScenarioCatalog.Normalize(config.map.scenarioId) == MapScenarioCatalog.DefaultId)
            pending.Add(new(SetupRequirementStage.Deployment, "objective." + config.objectiveId));
        if (!string.Equals(MapScenarioCatalog.Normalize(config.map.scenarioId), MapScenarioCatalog.DefaultId, StringComparison.Ordinal))
            pending.Add(new(SetupRequirementStage.Deployment, "scenario." + MapScenarioCatalog.Normalize(config.map.scenarioId)));

        foreach (var deployment in shipDeployments)
        {
            bool isVip = vipEntry is not null && deployment.fleetEntryInstanceId == "objective-escort-vip";
            bool isStation = stations.TryGetValue(deployment.fleetEntryInstanceId, out var stationEntry);
            var build = isVip || isStation ? new ShipBuildDto { instanceId = deployment.fleetEntryInstanceId, shipDefinitionId = (isVip ? vipEntry! : stationEntry!).Definition.ShipId } :
                players[deployment.ownerPlayerId].fleet.ships.Single(x => x.instanceId == deployment.fleetEntryInstanceId);
            var entry = isVip ? vipEntry! : isStation ? stationEntry! : shipsByDefinition[build.shipDefinitionId];
            var definition = entry.Definition;
            int index = shipStates.Count;
            // ShipId is the definition ID in the upstream state; distinct runtime identity lives in topology.
            shipStates.Add(new ShipRuntimeState(definition.ShipId, definition.Hull, definition.FrontShields,
                definition.LeftShields, definition.RightShields, definition.RearShields,
                Math.Clamp(deployment.startingSpeed, 0, definition.MaximumSpeed), definition.BraceTokens,
                definition.RedirectTokens, definition.EvadeTokens, entry.ContainTokens, entry.SalvoTokens, entry.ScatterTokens,
                definition.HasAuxiliaryHullZones ? definition.LeftAuxiliaryShields : -1,
                definition.HasAuxiliaryHullZones ? definition.RightAuxiliaryShields : -1));
            commands.Add(isStation ? CommandStackState.CreateWithoutCommands() : new CommandStackState(Math.Max(1, definition.Command)));
            var indices = new List<int>();
            foreach (string id in build.upgradeIds ?? Array.Empty<string>())
            {
                int upgradeIndex = upgradeStates.Count;
                var definitionCard = UpgradeCatalog.GetRequired(id);
                upgradeStates.Add(new UpgradeCardRuntimeState(definitionCard));
                upgradeTopology.Add(new(upgradeIndex, index, deployment.ownerPlayerId, definitionCard.id, false));
                indices.Add(upgradeIndex);
                if (config.mapMovementEnabled || id != "grav-shift-reroute")
                    AddUpgradeSetupRequirements(pending, id, deployment.ownerPlayerId, index, upgradeIndex, config.firstPlayerId);
            }
            shipTopology.Add(new(index, index, deployment.ownerPlayerId, deployment.runtimeUnitId,
                deployment.fleetEntryInstanceId, entry, indices.AsReadOnly(),
                definition.GameplayFootprint * TabletopMeasurements.MillimetresPerWorldUnit, isVip || isStation, isStation));
        }
        foreach (var deployment in squadronDeployments)
        {
            var build = players[deployment.ownerPlayerId].fleet.squadrons.Single(x => x.instanceId == deployment.fleetEntryInstanceId);
            var entry = squadronsByDefinition[build.squadronDefinitionId];
            var state = new SquadronRuntimeState();
            state.Initialize(entry.Definition.Hull, entry.Definition.BraceTokens, entry.Definition.ScatterTokens, entry.EvadeTokens);
            int squadronIndex = squadronStates.Count, entityIndex = shipStates.Count + squadronIndex;
            squadronStates.Add(state);
            int leaderIndex = -1;
            if (!string.IsNullOrWhiteSpace(build.leaderUpgradeId))
                throw new ArgumentException("Fighter-group cards have been removed from this engine.", nameof(match));
            squadronTopology.Add(new(entityIndex, squadronIndex, deployment.ownerPlayerId, deployment.runtimeUnitId,
                deployment.fleetEntryInstanceId, entry, leaderIndex,
                entry.Definition.GameplayFootprint * TabletopMeasurements.MillimetresPerWorldUnit));
        }
        foreach (var owner in upgradeTopology.Where(x => x.DefinitionId == "admiral-motti-commander").Select(x => x.Owner).Distinct())
            pending.Add(new(SetupRequirementStage.PostDeployment, "commander.motti-hull-bonus", owner));
        var registers = new MatchRegisterLayout(shipStates.Count + squadronStates.Count, upgradeStates.Count);
        string schema = ComputeSchema(config, data, registers);
        int deckSeed = StableDamageDeckSeed(config.matchId);
        var stateResult = new BattleState(schema, shipStates, squadronStates, commands,
            new[] { new DamageDeckState(deckSeed ^ 0x2C9277B5), new DamageDeckState(deckSeed ^ 0x5A17D39B) },
            upgradeStates, registers.Count, seed);
        var orderedDeployments = shipDeployments.Concat(squadronDeployments).ToArray();
        for (int index = 0; index < orderedDeployments.Length; index++)
        {
            var deployment = orderedDeployments[index];
            stateResult.Owners[index] = deployment.ownerPlayerId;
            stateResult.Poses[index] = new Pose(MatchTopology.ConvertVector(deployment.positionX, deployment.positionZ), deployment.yawDegrees);
        }
        foreach (var upgrade in upgradeTopology)
            stateResult.Registers[registers.UpgradeOwner(upgrade.UpgradeIndex)] = upgrade.InitialEntityIndex + 1;
        var placementTopology = new List<PreparedMapPlacement>();
        foreach (var placement in config.map.placements ?? Array.Empty<MapPlacementDto>())
        {
            int index = stateResult.SpatialObjects.Count;
            var size = placement.kind == MapPlacementKind.Obstacle ? MapObstacleGeometry.Size(placement)
                : MatchTopology.ConvertVector(placement.widthWorldUnits, placement.lengthWorldUnits);
            string shapeId = placement.kind == MapPlacementKind.Obstacle ? MapObstacleGeometry.Definition(placement).Id : string.Empty;
            if (placement.kind == MapPlacementKind.Obstacle)
            {
                placement.widthWorldUnits = size.X / TabletopMeasurements.MillimetresPerWorldUnit;
                placement.lengthWorldUnits = size.Y / TabletopMeasurements.MillimetresPerWorldUnit;
            }
            stateResult.SpatialObjects.Add(new SpatialObjectState((int)placement.kind,
                new Pose(MatchTopology.ConvertVector(placement.positionX, placement.positionZ), placement.yawDegrees),
                size, placement.ownerPlayerId, Payload: placement.startingSpeed, Flipped: placement.flipped, ShapeId: shapeId));
            placementTopology.Add(new(index, placement.placementId, placement.contentId, placement.displayName, placement.kind,
                TabletopMeasurements.FromWorldUnits(placement.interactionRangeWorldUnits), placement.startingSpeed, placement.useCustomSize));
        }
        stateResult.Frames.Add(new RuleFrame(MatchSetupRuleId));
        var topology = new MatchTopology(config, shipTopology.ToArray(), squadronTopology.ToArray(), upgradeTopology.ToArray(),
            placementTopology.ToArray(), registers);
        stateResult.Objective = PrepareObjective(config, topology);
        return new PreparedMatch(config, stateResult, topology, pending.ToArray());
    }

    public static int StableDamageDeckSeed(string value)
    {
        int seed = unchecked((int)2166136261);
        foreach (char character in value ?? string.Empty) seed = unchecked((seed ^ character) * 16777619);
        return seed;
    }

    private static IObjectiveRuntime? PrepareObjective(MatchConfigDto config, MatchTopology topology)
    {
        static OrientedRectangle ToRectangle(PreparedDeploymentZone zone) => new(
            new Pose((zone.MinimumMillimetres + zone.MaximumMillimetres) * .5f),
            zone.MaximumMillimetres - zone.MinimumMillimetres);
        string scenario = MapScenarioCatalog.Normalize(config.map.scenarioId);
        if (scenario == MapScenarioCatalog.DefaultId) return new OfficialObjectiveState(config.objectiveId, topology.EntityCount);
        if (scenario == MapScenarioCatalog.EscortVipId)
        {
            var vip = topology.Ships.Single(x => x.IsScenarioUnit);
            int interceptor = config.escortPlayerId == 1 ? 2 : 1;
            return new EscortVipObjective(vip.ShipIndex, config.escortPlayerId, interceptor,
                ToRectangle(topology.DeploymentZones.Single(x => x.Owner == interceptor)));
        }
        if (scenario == MapScenarioCatalog.SeizeStationId) return new SeizeStationObjective(Vector2.Zero);
        if (scenario == MapScenarioCatalog.IntelRetrievalId)
        {
            // BattlefieldLayoutView.CreateScenarioObjects positions the three numbered Intel tokens.
            float edge = TabletopMeasurements.FromWorldUnits(config.map.lengthWorldUnits * .5f - 2f);
            return new IntelRetrievalObjective(new[]
            {
                new IntelTokenState(0, new Vector2(0, edge)), new IntelTokenState(1, Vector2.Zero),
                new IntelTokenState(2, new Vector2(0, -edge))
            }, topology.DeploymentZones.ToDictionary(x => x.Owner, ToRectangle));
        }
        return null;
    }

    private static (ShipCatalogEntry Entry, UnitDeploymentDto Deployment) PrepareVip(MatchConfigDto config, FleetData data)
    {
        bool rebel = string.Equals(config.players.Single(x => x.playerId == config.escortPlayerId).fleet.faction,
            "rebel", StringComparison.OrdinalIgnoreCase);
        var presentationDefinition = data.Ships[rebel ? "gr75-medium-transports" : "gozanti-class-cruisers"];
        string id = rebel ? "objective-armored-gr75-vip-transport" : "objective-gozanti-vip-command-transport";
        var battery = new { redDice = 0, blueDice = 1, blackDice = 0 };
        // Exact domain arguments from BattleBootstrap.SpawnEscortVipTransport; no prefab dependency.
        var fields = JsonSerializer.SerializeToElement(new
        {
            shipId = id, displayName = rebel ? "Armored GR-75 VIP Transport" : "Gozanti VIP Command Transport",
            faction = rebel ? (int)ShipFaction.Rebel : (int)ShipFaction.Empire,
            shipType = rebel ? "gr75-rebel-transport" : "gozanti-class", points = 0, hull = 6,
            upgradeSlots = Array.Empty<string>(), frontShields = 3, leftShields = 2, rightShields = 2, rearShields = 2,
            hasAuxiliaryHullZones = false, leftAuxiliaryShields = 0, rightAuxiliaryShields = 0,
            command = 1, engineering = 3, squadron = 0, maximumSpeed = 3,
            braceTokens = 1, redirectTokens = 1, evadeTokens = 1,
            maneuverChart = new[] { new { jointYaw = new[] { 2 } }, new { jointYaw = new[] { 1, 2 } }, new { jointYaw = new[] { 0, 1, 2 } } },
            frontBattery = new { redDice = 1, blueDice = 2, blackDice = 0 },
            leftBattery = new { redDice = 1, blueDice = 1, blackDice = 0 },
            rightBattery = new { redDice = 1, blueDice = 1, blackDice = 0 }, rearBattery = battery,
            leftAuxiliaryBattery = new { redDice = 0, blueDice = 0, blackDice = 0 },
            rightAuxiliaryBattery = new { redDice = 0, blueDice = 0, blackDice = 0 },
            antiSquadronArmament = battery,
            gameplayFootprint = new { x = presentationDefinition.GameplayFootprint.X, y = presentationDefinition.GameplayFootprint.Y }
        });
        var definition = ShipDefinition.FromDefinitionFields(fields);
        var sourceZone = config.map.deploymentZones.Single(x => x.playerId == config.escortPlayerId);
        int interceptor = config.escortPlayerId == 1 ? 2 : 1;
        var targetZone = config.map.deploymentZones.Single(x => x.playerId == interceptor);
        var position = new Vector2((sourceZone.minimumX + sourceZone.maximumX) * .5f,
            (sourceZone.minimumZ + sourceZone.maximumZ) * .5f);
        var target = new Vector2((targetZone.minimumX + targetZone.maximumX) * .5f,
            (targetZone.minimumZ + targetZone.maximumZ) * .5f);
        float yaw = MathF.Atan2(target.X - position.X, target.Y - position.Y) * 57.29578f;
        var entry = new ShipCatalogEntry(id, definition, new ShipFleetMetadata("small", Array.Empty<string>()),
            0, 0, 0, "BattleBootstrap.SpawnEscortVipTransport");
        return (entry, new UnitDeploymentDto
        {
            runtimeUnitId = RuntimeUnitId.ForMatchEntry(config.matchId, config.escortPlayerId, RuntimeUnitKind.Ship, "objective-escort-vip").Value,
            fleetEntryInstanceId = "objective-escort-vip", ownerPlayerId = config.escortPlayerId, unitKind = RuntimeUnitKind.Ship,
            positionX = position.X, positionY = config.map.playPlaneY, positionZ = position.Y, yawDegrees = yaw, startingSpeed = 2
        });
    }

    private static FleetRuleset ResolveRuleset(string id) => id switch
    {
        "official" or "official-standard-400" => FleetRuleset.OfficialStandard400,
        "standard-400" => FleetRuleset.StandardPlay400,
        "standard-2025" => FleetRuleset.Standard2025,
        "custom" => FleetRuleset.Custom,
        "custom-upgrade-checklist" => FleetRuleset.CustomUpgradeChecklist,
        "fleet-sandbox" => FleetRuleset.Sandbox,
        _ => throw new NotSupportedException("Unrecognized source match ruleset: " + id)
    };

    private static void AddUpgradeSetupRequirements(List<PendingSetupRequirement> pending, string id, int owner, int entity, int upgrade, int firstPlayer)
    {
        id = id.ToLowerInvariant();
        string? deployment = id switch
        {
            "admiral-raddus-commander" or "profundity-title" or "rapid-launch-bays" => "deployment.reserve-unit",
            "hyperspace-rings" => "deployment.grant-scout",
            "g7-x-grav-well-projector" or "grav-shift-reroute" or "proximity-mines" => "deployment.place-setup-token",
            "invisible-hand-title" => "deployment.invisible-hand-reserve",
            "corvus-title" => "deployment.corvus-redeploy",
            "mercy-mission" => "setup.assign-mercy-mission",
            _ => null
        };
        if (deployment is not null) pending.Add(new(SetupRequirementStage.Deployment, deployment, owner, entity, upgrade));
        string? post = id switch
        {
            "garm-bel-iblis-commander" or "general-tagge-commander" or "shu-mai-officer" => "setup.choose-scheduled-rounds",
            "captain-needa-officer" or "vanguard-title" or "local-fire-control" or "kyrsta-agate-commander" => "setup.choose-defense-token",
            "taskmaster-grint-officer" => "setup.assign-command-token",
            "admiral-titus-officer" => "setup.choose-enemy-speed",
            "aspiration-title" => "setup.move-shields",
            "admiral-tarkin-commander" or "admiral-trench-commander" or "count-dooku-commander" or "general-draven-commander" or
                "grand-admiral-thrawn-commander" or "rune-haako-officer" or "resolute-title" => "setup.store-command-tokens",
            "gunnery-chief-varnillian" => owner != firstPlayer ? "setup.choose-varnillian-die" : "setup.store-varnillian-blank-red-die",
            "amity-title" or "concord-title" or "endeavor-title" or "tb-73-title" or "unity-title" or "executor-title" or
                "phoenix-home-title" or "relentless-title" or "nova-defiant-title" => "setup.fixed-ship-effects",
            _ => null
        };
        if (post is not null) pending.Add(new(SetupRequirementStage.PostDeployment, post, owner, entity, upgrade));
    }

    private static string ComputeSchema(MatchConfigDto config, FleetData data, MatchRegisterLayout registers)
    {
        var options = new JsonSerializerOptions { IncludeFields = true, MaxDepth = 64 };
        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(new
        {
            Version = SchemaVersion, Match = config, SourceUpgradeHash = UpgradeCatalog.SourceSha256,
            PlayerDecisionContract = "caller-discrete-v1", ObstacleFallbackContract = "caller-local-v1",
            MapMovementContract = config.mapMovementEnabled ? Engine.MapMovementRules.EnabledVariant : Engine.MapMovementRules.DisabledVariant,
            PhysicalObstacleShapesHash = ObstacleShapeCatalog.SourceSha256,
            Ships = data.ShipEntries, Squadrons = data.SquadronEntries, Objectives = data.Objectives,
            Upgrades = UpgradeCatalog.Cards, Registers = registers.Names
        }, options);
        return SchemaVersion + ":" + Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }
}
