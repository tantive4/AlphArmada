using System.Security.Cryptography;
using System.Text.Json;
using Armada.Core;
using Armada.Data;
using Armada.Fleet;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless.Catalog;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Search;
using Swarmada.Headless.Setup;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Worker;

public sealed record PreparedWorkerGame(GameSession Game, MatchConfigDto Match, MapSetupAudit MapAudit,
    int SetupActions, int SetupAttempt, string DeploymentPolicy, string SetupChoicePolicy,
    int[] FleetPoints, bool OfficialFleetValidation);

/// <summary>Caller-owned setup, using actual native deployment and setup rules.
/// A supplied match's authored deployment poses are authoritative; invalid poses
/// are reported, never moved silently by the engine or a tactical heuristic.</summary>
public sealed class ProductionGameSetup(WorkerOptions options, FleetData data)
{
    private readonly IFleetContentCatalog content = data.BuildFleetCatalog();
    public PreparedWorkerGame Create(string gameId, ulong seed, CancellationToken cancellationToken)
    {
        Exception? last = null;
        for (int attempt = 0; attempt < options.SetupRetries; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var match = options.Match is null ? BuiltIn(gameId) : JsonSerializer.Deserialize<MatchConfigDto>(File.ReadAllText(options.Match), WorkerOptions.Json)
                ?? throw new ArgumentException("Match JSON is empty.");
            if (options.Objective is not null) match.objectiveId = options.Objective;
            if (string.IsNullOrEmpty(match.objectiveId)) match.objectiveId = "targeting-beacons";
            if (!OfficialObjectives.SupportedIds.Contains(match.objectiveId, StringComparer.Ordinal))
                throw new ArgumentException("The worker needs one supported objective: " + match.objectiveId);
            if (match.mapMovementEnabled) throw new ArgumentException("The production worker uses the agreed static-map variant. Set mapMovementEnabled=false.");
            bool official = match.ruleset == FleetRuleset.OfficialStandard400.Id;
            var validator = new FleetLegalityValidator(content, official ? FleetRuleset.OfficialStandard400 : FleetRuleset.Sandbox,
                new FleetValidationOptions(reportGameplayCoverage: false));
            var reports = match.players.Select(player => validator.Validate(player.fleet)).ToArray();
            if (official && reports.Any(report => !report.IsLegal || report.TotalPoints > 400))
                throw new ArgumentException("Official worker fleet validation failed: " + string.Join("; ", reports.SelectMany(report => report.Issues)));
            ulong attemptSeed = SearchHash.Add(seed, (ulong)attempt);
            var game = new GameSession(MatchPreparation.Prepare(match, data, attemptSeed));
            int mapSeed = unchecked((int)SearchHash.Add(attemptSeed, "map"));
            var map = new MapSetupPolicy(mapSeed, options.SetupEpsilon, match);
            var random = new DeterministicRandom(SearchHash.Add(attemptSeed, "non-map-setup"));
            var scripted = ReadScript(); int scriptIndex = 0;
            try
            {
                for (int step = 0; step < options.SetupChoiceLimit; step++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var decision = game.GetDecision();
                    if (decision.Kind == DecisionKind.InvalidSetup) throw new InvalidSetupException(decision.InvalidSetup!);
                    if (game.Context.State.Registers[game.Context.Topology.Registers["setup.completed"]] != 0)
                    {
                        if (scriptIndex != scripted.Length) throw new ArgumentException("Unused actions remain in the supplied non-map setup script.");
                        return new(game, match, map.CaptureAudit(), step, attempt + 1,
                            options.Match is null ? "authored-worker-deployment-v1" : "supplied-match-deployment-poses",
                            scripted.Length == 0 ? "seeded-uniform-fixed-native-setup-v1" : "supplied-non-map-setup-action-script",
                            reports.Select(report => report.TotalPoints).ToArray(), official);
                    }
                    if (map.TryApplyNext(game)) continue;
                    if (decision.Kind == DecisionKind.Chance) { game.SampleChance(); continue; }
                    if (scriptIndex < scripted.Length) { game.Apply(scripted[scriptIndex++]); continue; }
                    var deployment = decision.Actions.FirstOrDefault(option => option.Template.Kind == GameActionKind.Deploy);
                    if (deployment is not null)
                    {
                        int entity = deployment.Template.Source; var pose = game.Context.State.Poses[entity];
                        game.Apply(deployment.Template with { Position = pose.Position,
                            YawDegrees = deployment.AcceptsYaw ? pose.YawDegrees : deployment.Template.YawDegrees,
                            Speed = game.Context.IsShip(entity) ? game.Context.State.Ships[entity].CurrentSpeed : deployment.Template.Speed });
                        continue;
                    }
                    var fixedActions = decision.Actions.Where(option => !option.AcceptsPosition && !option.AcceptsYaw && !option.AcceptsCourse && !option.AcceptsFlip).ToArray();
                    if (fixedActions.Length == 0)
                        throw new SearchCoverageException("A non-map physical setup choice requires a supplied setup action: " +
                            (game.Context.Flow.Continuations.LastOrDefault()?.Rule ?? decision.Phase.ToString()));
                    game.Apply(fixedActions[random.NextInt(fixedActions.Length)].Template);
                }
                throw new WorkerLimitException("setup-choice-limit", "Setup did not finish within its explicit choice bound.");
            }
            catch (InvalidSetupException error) when (error.Result.Retryable && options.Match is null && attempt + 1 < options.SetupRetries) { last = error; }
            catch (MapSetupCoverageException error) when (options.Match is null && attempt + 1 < options.SetupRetries) { last = error; }
        }
        throw new SearchCoverageException("No valid generated map setup was found within the bounded retry policy: " + last?.Message);
    }

    private GameAction[] ReadScript()
    {
        if (options.Setup is null) return [];
        using var document = JsonDocument.Parse(File.ReadAllText(options.Setup));
        var actions = document.RootElement.ValueKind == JsonValueKind.Array ? document.RootElement : document.RootElement.GetProperty("actions");
        return actions.Deserialize<GameAction[]>(WorkerOptions.Json) ?? [];
    }

    private MatchConfigDto BuiltIn(string gameId)
    {
        var first = FleetDraft.Create("Worker Rebel", "rebel", 400, "custom");
        var second = FleetDraft.Create("Worker Empire", "empire", 400, "custom");
        if (options.Preset == "small")
        {
            first.AddShip("cr90-corvette-a"); first.AddSquadron("x-wing-squadron");
            second.AddShip("raider-i-class-corvette"); second.AddSquadron("tie-fighter-squadron");
        }
        else if (options.Preset == "mixed")
        {
            first.AddShip("assault-frigate-mark-ii-b", ["general-dodonna-commander", "electronic-countermeasures"]);
            first.AddShip("cr90-corvette-a", ["comms-net"]);
            foreach (string squad in new[] { "luke-skywalker-x-wing-squadron", "x-wing-squadron", "x-wing-squadron", "a-wing-squadron" }) first.AddSquadron(squad);
            second.AddShip("imperial-ii-class-star-destroyer", ["admiral-motti-commander", "leading-shots"]);
            second.AddShip("raider-i-class-corvette", ["ordnance-experts"]);
            foreach (string squad in new[] { "darth-vader-tie-advanced-squadron", "tie-fighter-squadron", "tie-fighter-squadron", "tie-interceptor-squadron" }) second.AddSquadron(squad);
        }
        else if (options.Preset == "fleet400")
        {
            first.AddShip("assault-frigate-mark-ii-b", ["general-dodonna-commander", "intel-officer", "gunnery-team", "expanded-hangar-bay", "electronic-countermeasures", "xi7-turbolasers"]);
            first.AddShip("mc30c-torpedo-frigate", ["ordnance-experts", "external-racks", "electronic-countermeasures", "admonition-title"]);
            first.AddShip("cr90-corvette-a", ["turbolaser-reroute-circuits", "jainas-light-title"]);
            foreach (string squad in new[] { "luke-skywalker-x-wing-squadron", "dutch-vander-y-wing-squadron", "x-wing-squadron", "x-wing-squadron", "a-wing-squadron", "a-wing-squadron", "y-wing-squadron", "jan-ors-moldy-crow" }) first.AddSquadron(squad);
            second.AddShip("imperial-ii-class-star-destroyer", ["admiral-sloane-commander", "gunnery-team", "early-warning-system", "leading-shots", "xi7-turbolasers"]);
            second.AddShip("quasar-fire-i-class-cruiser-carrier", ["flight-controllers", "expanded-hangar-bay", "squall-title"]);
            second.AddShip("gozanti-class-cruisers", ["comms-net"]);
            foreach (string squad in new[] { "maarek-stele-tie-defender-squadron", "colonel-jendon-lambda-class-shuttle", "howlrunner-tie-fighter-squadron", "tie-fighter-squadron", "tie-fighter-squadron", "tie-fighter-squadron", "tie-interceptor-squadron", "tie-interceptor-squadron" }) second.AddSquadron(squad);
        }
        else throw new ArgumentException("Unknown built-in preset; use small, mixed, fleet400 or --match with a supplied match JSON.");
        var match = DefaultMatchFactory.CreateLocal(first.Document, second.Document, content: data, objectiveId: options.Objective ?? "targeting-beacons");
        match.ruleset = options.Preset == "fleet400" ? FleetRuleset.OfficialStandard400.Id : FleetRuleset.Sandbox.Id; match.mapMovementEnabled = false;
        // Stable match identity; preserve each official factory-authored pose.
        match.matchId = gameId;
        foreach (var deployment in match.deployments)
        {
            deployment.runtimeUnitId = RuntimeUnitId.ForMatchEntry(gameId, deployment.ownerPlayerId,
                deployment.unitKind, deployment.fleetEntryInstanceId).Value;
            // Source ship poses are retained. A neutral authored squadron row is
            // moved two world units toward board center, clear of large carriers'
            // front edges while remaining within the legal deployment distance.
            // This preset construction never changes a supplied match document.
            if (deployment.unitKind == RuntimeUnitKind.Squadron) deployment.positionZ += deployment.ownerPlayerId == 1 ? 2 : -2;
        }
        return match;
    }
}
