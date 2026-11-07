using System.Diagnostics;
using System.Text.Json;
using Armada.Core;
using Armada.Data;
using Armada.Fleet;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless;
using Swarmada.Headless.Catalog;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Setup;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Runtime;

internal static class Program
{
    private static readonly JsonSerializerOptions Json = new()
    { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, IncludeFields = true };
    private static FleetData? fleetData;
    private static ImmutableFleetContentCatalog? fleetCatalog;
    private static readonly Dictionary<long, GameSession> games = new();
    private static readonly Dictionary<long, DiscreteActionAdapter> discreteAdapters = new();
    private static long nextGameId;
    public static int Main(string[] args)
    {
        try
        {
            string operation = args.Length == 0 ? "status" : args[0];
            string? data = null;
            if (args.Length > 1)
            {
                if (args.Length != 3 || args[1] != "--data")
                    throw new ArgumentException("Usage: status|serve|benchmark [--data <catalog-directory>]");
                data = Path.GetFullPath(args[2]);
                UpgradeCatalog.LoadJson(Path.Combine(data, "official-upgrades.json"));
                fleetData = FleetData.LoadDirectory(data);
                fleetCatalog = fleetData.BuildFleetCatalog();
            }
            if (operation == "serve") return Serve(data);
            object result = operation switch
            {
                "status" => Status(data),
                "benchmark" => Benchmark(100000),
                _ => throw new ArgumentException("Unknown CLI operation: " + operation)
            };
            Console.WriteLine(JsonSerializer.Serialize(result, Json)); return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }

    private static object Status(string? data) => new
    {
        protocolVersion = 1, upstreamVersion = PortStatus.UpstreamVersion,
        headlessMatchReady = PortStatus.HeadlessMatchReady, branchReplayReady = PortStatus.BranchReplayReady,
        structuredEncoderReady = PortStatus.StructuredEncoderReady,
        fullGameReady = PortStatus.FullGameReady, trainingReady = PortStatus.TrainingReady,
        portableComponents = new[] { "deployment", "commands", "rounds", "combat", "damage-cards", "upgrade-timings",
            "movement", "squadrons", "reserves", "source-scenarios", "deterministic-chance", "reusable-search-positions" },
        loadedUpgradeCount = UpgradeCatalog.IsLoaded ? UpgradeCatalog.Cards.Count : 0,
        loadedShips = fleetData?.ShipEntries.Count ?? 0,
        loadedSquadrons = fleetData?.SquadronEntries.Count ?? 0,
        loadedObjectives = fleetData?.Objectives.Count ?? 0,
        executableOfficialObjectives = PortStatus.ExecutableOfficialObjectiveCount,
        callerOwnedDiscreteActions = true, automaticObstaclePlacementOptimization = false,
        mapMovementEnabledByDefault = false, defaultRulesVariant = MapMovementRules.DisabledVariant,
        dataDirectory = data, remainingGameplayWork = PortStatus.RemainingGameplayWork
    };

    private static int Serve(string? data)
    {
        string? line;
        while ((line = Console.ReadLine()) is not null)
        {
            long? id = null;
            try
            {
                using var document = JsonDocument.Parse(line);
                var root = document.RootElement;
                if (root.TryGetProperty("id", out var requestId)) id = requestId.GetInt64();
                string op = root.GetProperty("op").GetString() ?? throw new ArgumentException("Operation required.");
                object result = op switch
                {
                    "status" => Status(data),
                    "coverage" => Coverage(),
                    "attack_batch" => ResolveBatch(root.GetProperty("requests")),
                    "validate_fleet" => ValidateFleet(root),
                    "starter_fleet" => FleetTemplateFactory.CreateStarter(RequiredFleetCatalog(),
                        root.GetProperty("faction").GetString(), "Headless starter").Document,
                    "import_fleet" => new FleetTextImporter(RequiredFleetCatalog()).Import(root.GetProperty("text").GetString()),
                    "default_match" => DefaultMatchFactory.CreateLocal(
                        RequiredFleet(root.GetProperty("playerOneFleet")), RequiredFleet(root.GetProperty("playerTwoFleet")),
                        firstPlayerId: root.TryGetProperty("firstPlayerId", out var initiative) ? initiative.GetInt32() : 1,
                        content: fleetData ?? throw new InvalidOperationException("Load catalogs first."),
                        objectiveId: root.TryGetProperty("objectiveId", out var objective) ? objective.GetString() ?? string.Empty : "targeting-beacons"),
                    "prepare_match" => PrepareMatch(root),
                    "benchmark" => Benchmark(root.TryGetProperty("iterations", out var count) ? count.GetInt32() : 100000),
                    "create_game" => CreateGame(root),
                    "game_decision" => GameView(RequiredGame(root)),
                    "game_state" => RequiredGame(root).CaptureSnapshot(),
                    "game_apply" => ApplyGame(root),
                    "game_discrete_decision" => DiscreteGameDecision(root),
                    "game_apply_discrete" => ApplyDiscreteGame(root),
                    "game_chance" => ChanceGame(root),
                    "game_clone" => CloneGame(root),
                    "game_restore" => RestoreGame(root),
                    "game_close" => CloseGame(root),
                    "game_benchmark" => GameplayBenchmark.Run(RequiredGame(root),
                        root.TryGetProperty("path", out var path) ? path.Deserialize<GameAction[]>(Json) ?? [] : [],
                        root.TryGetProperty("iterations", out var branchCount) ? branchCount.GetInt32() : 10_000),
                    "game_time_limit" => TimeLimitGame(root),
                    "game_concede" => ConcedeGame(root),
                    _ => throw new ArgumentException("Unknown protocol operation: " + op)
                };
                Console.WriteLine(JsonSerializer.Serialize(new { id, ok = true, result }, Json));
            }
            catch (InvalidSetupException error)
            {
                Console.WriteLine(JsonSerializer.Serialize(new
                { id, ok = false, error = new { type = nameof(InvalidSetupException), message = error.Message, invalidSetup = error.Result } }, Json));
            }
            catch (Exception error) when (error is JsonException or ArgumentException or InvalidOperationException or
                KeyNotFoundException or NotSupportedException or OverflowException or FormatException)
            {
                Console.WriteLine(JsonSerializer.Serialize(new
                { id, ok = false, error = new { type = error.GetType().Name, message = error.Message } }, Json));
            }
            Console.Out.Flush();
        }
        return 0;
    }

    private static ImmutableFleetContentCatalog RequiredFleetCatalog() => fleetCatalog ??
        throw new InvalidOperationException("Load the asset-derived catalogs with --data first.");
    private static object Coverage() => new
    {
        fullGameReady = PortStatus.FullGameReady,
        headlessMatchReady = PortStatus.HeadlessMatchReady,
        note = "Handler ownership is an audit aid; it is not a claim that every interaction has passed full-game parity validation.",
        upgrades = UpgradeCatalog.Cards.Select(card => new { card.id, card.name, card.slot,
            upstreamImplemented = OfficialContentPolicy.IsGameplayImplemented(card.id, card.publisher),
            declaredPortHandler = CombatModule.SupportsUpgrade(card.id) }).ToArray(),
        executableScenarios = new[] { "default", "escort-the-vip", "intel-retrieval", "seize-the-station" },
        executableObjectives = OfficialObjectives.SupportedIds
    };
    private static object CreateGame(JsonElement root)
    {
        var config = root.GetProperty("match").Deserialize<MatchConfigDto>(Json) ?? throw new ArgumentException("Missing match document.");
        ulong seed = root.TryGetProperty("seed", out var chosenSeed) ? chosenSeed.GetUInt64() : 0;
        var prepared = MatchPreparation.Prepare(config, fleetData ?? throw new InvalidOperationException("Load catalogs first."), seed);
        var game = new GameSession(prepared); long gameId = checked(++nextGameId);
        games.Add(gameId, game);
        try { return new { gameId, topology = prepared.Topology, view = GameView(game), fullGameReady = PortStatus.FullGameReady }; }
        catch { games.Remove(gameId); throw; }
    }
    private static GameSession RequiredGame(JsonElement root) => games.TryGetValue(root.GetProperty("gameId").GetInt64(), out var game) ? game :
        throw new KeyNotFoundException("Unknown or closed game handle.");
    private static object GameView(GameSession game) => new { decision = game.GetDecision(), round = game.Context.State.Round, revision = game.Context.State.Revision,
        mapMovementEnabled = MapMovementRules.Enabled(game.Context), rulesVariant = MapMovementRules.Variant(game.Context),
        setupComplete = game.Context.State.Registers[game.Context.Topology.Registers["setup.completed"]] != 0,
        invalidSetup = game.InvalidSetup,
        mercyMissionReports = game.Context.State.Outcome is null ? null : MapAbilityEffects.GetMercyMissionAwards(game.Context),
        tournamentResult = game.Context.State.Outcome is null ? null : TournamentRules.GetResult(game.Context) };
    private static object TimeLimitGame(JsonElement root)
    { var game = RequiredGame(root); if (game.InvalidSetup is { } invalid) throw new InvalidSetupException(invalid); InvalidateDiscrete(root); TournamentRules.RequestTimeLimit(game.Context); return GameView(game); }
    private static object ConcedeGame(JsonElement root)
    { var game = RequiredGame(root); if (game.InvalidSetup is { } invalid) throw new InvalidSetupException(invalid); InvalidateDiscrete(root); TournamentRules.Concede(game.Context, root.GetProperty("player").GetInt32()); return GameView(game); }
    private static object ApplyGame(JsonElement root)
    {
        var game = RequiredGame(root);
        var action = root.GetProperty("action").Deserialize<GameAction>(Json) ?? throw new ArgumentException("Missing gameplay action.");
        InvalidateDiscrete(root);
        game.Apply(action); return GameView(game);
    }
    private static DiscreteActionAdapter DiscreteAdapter(JsonElement root)
    {
        long id = root.GetProperty("gameId").GetInt64();
        if (!discreteAdapters.TryGetValue(id, out var adapter)) discreteAdapters[id] = adapter = new();
        return adapter;
    }
    private static void InvalidateDiscrete(JsonElement root)
    {
        if (discreteAdapters.TryGetValue(root.GetProperty("gameId").GetInt64(), out var adapter)) adapter.Invalidate();
    }
    private static object DiscreteGameDecision(JsonElement root)
    {
        var game = RequiredGame(root);
        var candidates = root.GetProperty("candidates").Deserialize<DiscreteActionCandidate[]>(Json)
            ?? throw new ArgumentException("Missing finite discrete candidates.");
        var view = GameView(game);
        var decision = DiscreteAdapter(root).GetDecision(game, candidates);
        return new { view, discreteDecision = decision };
    }
    private static object ApplyDiscreteGame(JsonElement root)
    {
        var game = RequiredGame(root);
        DiscreteAdapter(root).Apply(game, root.GetProperty("actionId").GetInt32(), root.GetProperty("decisionToken").GetInt64());
        return GameView(game);
    }
    private static object ChanceGame(JsonElement root)
    { var game = RequiredGame(root); InvalidateDiscrete(root); var action = game.SampleChance(); return new { action, view = GameView(game) }; }
    private static object CloneGame(JsonElement root)
    {
        var clone = RequiredGame(root).DeepClone(); long gameId = checked(++nextGameId); games.Add(gameId, clone);
        try { return new { gameId, view = GameView(clone) }; }
        catch { games.Remove(gameId); throw; }
    }
    private static object RestoreGame(JsonElement root)
    {
        var game = RequiredGame(root);
        var snapshot = root.GetProperty("snapshot").Deserialize<GameSnapshot>(Json) ?? throw new ArgumentException("Missing game snapshot.");
        InvalidateDiscrete(root);
        game.RestoreSnapshot(snapshot); return GameView(game);
    }
    private static object CloseGame(JsonElement root)
    {
        long id = root.GetProperty("gameId").GetInt64(); discreteAdapters.Remove(id);
        return new { closed = games.Remove(id) };
    }
    private static FleetListDto RequiredFleet(JsonElement element) => element.Deserialize<FleetListDto>(Json) ??
        throw new ArgumentException("Missing fleet document.");
    private static object PrepareMatch(JsonElement root)
    {
        var config = root.GetProperty("match").Deserialize<MatchConfigDto>(Json) ??
            throw new ArgumentException("Missing match document.");
        ulong seed = root.TryGetProperty("seed", out var chosenSeed) ? chosenSeed.GetUInt64() : 0;
        var prepared = MatchPreparation.Prepare(config, fleetData ??
            throw new InvalidOperationException("Load the asset-derived catalogs first."), seed);
        return new { scope = "Initial state only; pending gameplay setup and battle handlers are not executed.",
            topology = prepared.Topology, state = prepared.InitialState.CaptureSnapshot(),
            pendingSetupRequirements = prepared.PendingSetupRequirements };
    }
    private static object ValidateFleet(JsonElement root)
    {
        var fleet = RequiredFleet(root.GetProperty("fleet"));
        string rulesId = root.TryGetProperty("ruleset", out var selected) ?
            selected.GetString() ?? "official-standard-400" : "official-standard-400";
        var rules = rulesId switch
        {
            "official-standard-400" => FleetRuleset.OfficialStandard400,
            "standard-400" => FleetRuleset.StandardPlay400,
            "standard-2025" => FleetRuleset.Standard2025,
            "custom" => FleetRuleset.Custom,
            _ => throw new ArgumentException("Unknown fleet ruleset: " + rulesId)
        };
        var report = new FleetLegalityValidator(RequiredFleetCatalog(), rules,
            new FleetValidationOptions(reportGameplayCoverage: false)).Validate(fleet);
        return new { scope = "Upstream fleet legality only; this does not establish headless gameplay coverage.", report };
    }

    private static object[] ResolveBatch(JsonElement requests)
    {
        if (requests.ValueKind != JsonValueKind.Array || requests.GetArrayLength() > 4096)
            throw new ArgumentException("requests must be an array of at most 4096 attacks.");
        var result = new object[requests.GetArrayLength()];
        int i = 0;
        foreach (var request in requests.EnumerateArray())
        {
            int red = RequiredInt(request, "redDice"), blue = RequiredInt(request, "blueDice"),
                black = RequiredInt(request, "blackDice"), seed = RequiredInt(request, "seed");
            float distance = request.GetProperty("distanceMm").GetSingle();
            if (red < 0 || blue < 0 || black < 0 || red + (long)blue + black > 1024 ||
                !float.IsFinite(distance) || distance < 0)
                throw new ArgumentException("Invalid dice counts or millimetre distance.");
            var attack = AttackResolver.Resolve(new AttackRequest(red, blue, black,
                TabletopMeasurements.ToWorldUnits(distance), seed,
                OptionalInt(request, "rerolls"), OptionalInt(request, "blackRerolls"),
                diceAlreadyGathered: request.TryGetProperty("diceAlreadyGathered", out var gathered) && gathered.GetBoolean()));
            result[i++] = new { attack.Range, attack.IsLegal, attack.DiceRolled, attack.Hits,
                attack.Criticals, attack.Accuracies, attack.Blanks, attack.Damage, attack.Dice };
        }
        return result;
    }
    private static int RequiredInt(JsonElement element, string name) => element.GetProperty(name).GetInt32();
    private static int OptionalInt(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) ? value.GetInt32() : 0;

    private static object Benchmark(int iterations)
    {
        if (iterations < 1 || iterations > 10000000) throw new ArgumentOutOfRangeException(nameof(iterations));
        for (int i = 0; i < 10000; i++) AttackResolver.Resolve(new(2, 2, 2, 1f, i));
        long checksum = 0, before = GC.GetAllocatedBytesForCurrentThread();
        var watch = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++) checksum += AttackResolver.Resolve(new(2, 2, 2, 1f, i)).Damage;
        watch.Stop();
        long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
        var ship = new ShipRuntimeState("snapshot-benchmark", 8, 3, 2, 2, 1, 2);
        int snapshots = Math.Min(iterations, 10000);
        var snapshotWatch = Stopwatch.StartNew();
        for (int i = 0; i < snapshots; i++)
        {
            var snapshot = ship.CaptureSnapshot(); ship.ApplyDamage(HullZone.Front, 1); ship.RestoreSnapshot(snapshot);
        }
        snapshotWatch.Stop();
        return new
        {
            scope = "Microbenchmarks only; excludes full gameplay, legal-action generation, MCTS and inference.",
            iterations, seconds = watch.Elapsed.TotalSeconds,
            attackResolutionsPerSecond = iterations / watch.Elapsed.TotalSeconds,
            allocatedBytesPerAttack = bytes / (double)iterations, checksum,
            shipCaptureMutationRestorePerSecond = snapshots / snapshotWatch.Elapsed.TotalSeconds
        };
    }
}
