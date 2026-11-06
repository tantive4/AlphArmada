using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Defense objectives, including deployment/reserves, neutral attacks and command-value scoring.</summary>
public static class OfficialDefenseObjectives
{
    private const string Prefix = "objective.defense.";
    private const float Epsilon = .0115f;
    private static OfficialObjectiveState State(GameContext game) => OfficialObjectives.State(game)!;
    private static int First(GameContext game) => OfficialObjectives.FirstPlayer(game);
    private static int Second(GameContext game) => OfficialObjectives.SecondPlayer(game);
    private static int Anchor(GameContext game, int owner) => game.Topology.Ships.First(s => !s.IsScenarioUnit && s.Owner == owner).EntityIndex;
    public static bool Handles(string rule) => rule.StartsWith(Prefix, StringComparison.Ordinal);
    public static void ResolveUndeployableSquadrons(GameContext game)
    {
        if (!OfficialObjectives.Is(game, "hyperspace-assault") || OfficialObjectives.LiveShips(game, Second(game)).Any()) return;
        foreach (var squadron in game.Topology.Squadrons.Where(s => s.Owner == Second(game) && !game.IsReserved(s.EntityIndex)))
        {
            game.State.Squadrons[squadron.SquadronIndex].DestroyWithoutDamage();
            game.State.Registers[game.Topology.Registers[$"entity.{squadron.EntityIndex}.placed"]] = 1;
        }
    }
    public static IEnumerable<GameContinuation> SetupChoices(GameContext game, bool afterDeployment)
    {
        string id = OfficialObjectives.Id(game);
        if (!afterDeployment)
        {
            if (id is "capture-the-vip" or "fire-lanes" or "hyperspace-assault" or "planetary-ion-cannon")
            {
                if (id == "hyperspace-assault") yield return new(Prefix + "hyperspace-reserve", Anchor(game, Second(game)));
                yield return new(Prefix + "place-tokens", Anchor(game, Second(game)), Text: id);
                if (id == "fire-lanes" && MapMovementRules.Enabled(game)) yield return new(Prefix + "fire-lanes-adjust", Anchor(game, First(game)));
            }
            if (id == "fighter-ambush")
                foreach (var squadron in game.Topology.Squadrons.Where(s => s.Owner == Second(game) && !game.IsReserved(s.EntityIndex))) Reserve(game, squadron.EntityIndex);
            yield break;
        }
        if (id == "abandoned-mining-facility")
            foreach (int ship in OfficialObjectives.LiveShips(game, Second(game))) TokenAssignments.TryAssign(game, ship, ShipCommand.Engineering);
        if (id == "fleet-in-being")
            foreach (int ship in OfficialObjectives.LiveShips(game)) OfficialObjectives.SetTokens(game, ship,
                game.State.Owners[ship] == First(game) ? 1 : game.Topology.Ships[ship].CatalogEntry.Definition.Command);
        if (id == "fighter-ambush") yield return new(Prefix + "fighter-deploy", Anchor(game, Second(game)));
        if (id == "jamming-barrier") yield return new(Prefix + "place-tokens", Anchor(game, Second(game)), Text: id);
        if (id == "rift-ambush") yield return new(Prefix + "rift-setup", Anchor(game, Second(game)));
    }
    public static IEnumerable<GameContinuation> TimingChoices(GameContext game, string timing, int source, int target)
    {
        string id = OfficialObjectives.Id(game);
        if (timing == "attack-declared" && id == "fleet-in-being" && game.IsShip(target) && OfficialObjectives.TokenCount(game, target) > 0 &&
            Enum.GetValues<DefenseTokenType>().Any(type => game.State.Ships[target].GetDefenseTokenInstances(type).Any(token => token.Exhausted)))
            yield return new(Prefix + "fleet-ready", target);
        if (timing == "end-command-phase" && id == "planetary-ion-cannon" && PlanetaryTargets(game).Any())
            yield return new(Prefix + "planetary", Anchor(game, Second(game)));
        if (timing == "start-round" && id == "hyperspace-assault" && game.State.Round > 1 && State(game).ReservedEntities.Any(game.IsReserved))
            yield return new(Prefix + "hyperspace-round", Anchor(game, Second(game)));
        if (timing == "after-maneuver" && id == "rift-ambush" && game.IsShip(source) && OfficialObjectives.Live(game, source) &&
            game.Flow.Flags.GetValueOrDefault("activation.in-progress") == source + 1 &&
            State(game).Flags.GetValueOrDefault("rift.used." + source) != game.Flow.Flags.GetValueOrDefault("activation.serial") && AtRift(game, source))
            yield return new(Prefix + "rift-maneuver", source);
        if (timing == "end-round")
        {
            if (id == "contested-outpost") yield return new(Prefix + "outpost-score");
            if (id == "fire-lanes") yield return new(Prefix + "fire-lanes-score");
            if (id == "fleet-in-being") yield return new(Prefix + "fleet-remove");
        }
    }
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        if (OfficialObjectives.Is(game, "capture-the-vip") && timing == "command-reveal" && game.IsShip(source) && State(game).Flags.GetValueOrDefault("vip.carrier") == 0)
            foreach (int token in OfficialObjectives.TokenIndices(game).Where(i => OfficialObjectives.DistanceToPiece(game, source, i) <= TabletopMeasurements.DistanceOne + Epsilon))
                yield return new(new(GameActionKind.UseAbility, source, token, AbilityId: Prefix + "vip-collect"), "Pick up the VIP");
    }
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (action.AbilityId != Prefix + "vip-collect") return false;
        if (!GetActions(game, window.Text, window.Source, window.Target).Any(o => GameSession.Matches(o, action))) throw new ArgumentException("The VIP is not available.");
        OfficialObjectives.RemoveToken(game, action.Target); State(game).Flags["vip.carrier"] = action.Source + 1;
        OfficialObjectives.SetTokens(game, action.Source, 1); return true;
    }
    public static void Observe(GameContext game)
    {
        if (!OfficialObjectives.Is(game, "capture-the-vip") || State(game).Flags.GetValueOrDefault("vip.carrier") is not > 0) return;
        int carrier = (int)State(game).Flags["vip.carrier"] - 1;
        if (!game.IsDestroyed(carrier)) return;
        State(game).Flags["vip.carrier"] = 0; OfficialObjectives.SetTokens(game, carrier, 0);
        // The fixed-map rules variant removes the carried VIP when its carrier
        // is destroyed; it does not introduce a new coordinate placement choice.
        if (MapMovementRules.Enabled(game))
            OfficialObjectives.QueueChoices(game, [new(Prefix + "vip-drop", carrier, 3 - game.State.Owners[carrier])], game.Flow.Phase);
    }
    public static void ObserveAttack(GameContext game, AttackState attack)
    {
        if (OfficialObjectives.Is(game, "fighter-ambush") && !attack.EnvironmentalDamage && attack.Attacker >= game.State.Ships.Length &&
            game.IsShip(attack.Defender) && attack.HullCardsDealt > 0 && !attack.EffectFlags.ContainsKey("objective.fighter-ambush.scored"))
        { attack.EffectFlags["objective.fighter-ambush.scored"] = 1; OfficialObjectives.AwardVictory(game, game.State.Owners[attack.Attacker]); }
    }
    public static int ScoreBonus(GameContext game, int player)
    {
        if (OfficialObjectives.Is(game, "capture-the-vip") && State(game).Flags.GetValueOrDefault("vip.carrier") is > 0)
        { int carrier = (int)State(game).Flags["vip.carrier"] - 1; return OfficialObjectives.Live(game, carrier) && game.State.Owners[carrier] == player ? 50 : 0; }
        if (OfficialObjectives.Is(game, "fleet-in-being")) return OfficialObjectives.LiveShips(game, 3 - player).Count(s => OfficialObjectives.TokenCount(game, s) == 0) * 15;
        return 0;
    }
    public static IEnumerable<GameActionOption> GetEngineeringActions(GameContext game, int ship)
    {
        if (!OfficialObjectives.Is(game, "abandoned-mining-facility")) yield break;
        int points = game.Flow.ShipActivations[ship].EngineeringPoints;
        foreach (int obstacle in OfficialObjectives.Obstacles(game))
        {
            string kind = OfficialObjectives.ObstacleId(game, obstacle); if (kind is not ("station" or "dust-field")) continue;
            if (OfficialObjectives.DistanceToPiece(game, ship, obstacle) > TabletopMeasurements.DistanceOne + Epsilon) continue;
            if (State(game).Flags.GetValueOrDefault("mining.round." + ship) == game.State.Round && State(game).Flags.GetValueOrDefault("mining.source." + ship) != obstacle + 1) continue;
            int cost = kind == "station" ? 3 : 2;
            for (int count = 1; count <= points / cost; count++)
                yield return new(new(GameActionKind.UseAbility, ship, Index: obstacle, Amount: count, AbilityId: Prefix + "mine-engineering"), $"Spend {count * cost} engineering to gain {count} victory tokens");
        }
    }
    public static bool TryApplyEngineering(GameContext game, GameAction action)
    {
        if (action.AbilityId != Prefix + "mine-engineering") return false;
        if (!GetEngineeringActions(game, action.Source).Any(o => GameSession.Matches(o, action))) throw new ArgumentException("Illegal mining Engineering action.");
        string kind = OfficialObjectives.ObstacleId(game, action.Index); int cost = kind == "station" ? 3 : 2;
        game.Flow.ShipActivations[action.Source].EngineeringPoints -= cost * action.Amount;
        State(game).Flags["mining.round." + action.Source] = game.State.Round; State(game).Flags["mining.source." + action.Source] = action.Index + 1;
        OfficialObjectives.AwardVictory(game, game.State.Owners[action.Source], action.Amount);
        string gainedKey = "mining.gained." + action.Source;
        State(game).Flags[gainedKey] = State(game).Flags.GetValueOrDefault("mining.previous-round." + action.Source) == game.State.Round
            ? State(game).Flags.GetValueOrDefault(gainedKey) + action.Amount : action.Amount;
        State(game).Flags["mining.previous-round." + action.Source] = game.State.Round;
        if (kind == "dust-field" && State(game).Flags[gainedKey] > 1)
        {
            game.State.SpatialObjects[action.Index] = game.State.SpatialObjects[action.Index] with { Active = false };
            TokenObstacleAttachments.RemoveObstacle(game, action.Index);
        }
        return true;
    }
    public static bool IgnoresObstacleDamage(GameContext game, int entity, int obstacle) => OfficialObjectives.Is(game, "asteroid-tactics") &&
        game.State.Owners[entity] == Second(game) && OfficialObjectives.ObstacleId(game, obstacle).StartsWith("asteroid", StringComparison.Ordinal);
    public static IEnumerable<GameContinuation> ObstacleChoices(GameContext game, int entity, int obstacle, bool effectsIgnored)
    {
        if (!effectsIgnored && OfficialObjectives.Is(game, "asteroid-tactics") && OfficialObjectives.Live(game, entity) &&
            OfficialObjectives.ObstacleId(game, obstacle).StartsWith("asteroid", StringComparison.Ordinal) &&
            (game.IsShip(entity) || game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition.IsUnique))
            yield return new(Prefix + "asteroid-recover", entity);
    }
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        if (frame.Rule is (Prefix + "fire-lanes-adjust" or Prefix + "vip-drop") && !MapMovementRules.Enabled(game))
            return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass, frame.Source), "Skip disabled objective token relocation")]);
        var options = new List<GameActionOption>(); int owner = frame.Source >= 0 ? game.State.Owners[frame.Source] : Second(game);
        void Add(GameAction action, string text, bool position = false, bool yaw = false) => options.Add(new(action, text, position, yaw));
        switch (frame.Rule)
        {
            case Prefix + "place-tokens": Add(new(GameActionKind.PlaceToken, frame.Source), "Place objective token", true); break;
            case Prefix + "fire-lanes-adjust":
                if (MapMovementRules.Enabled(game) && frame.Stage < 3) Add(new(GameActionKind.PlaceToken, frame.Source, Target: Tokens(game).ElementAt(frame.Stage)), "Move this Fire Lanes token", true);
                Add(new(GameActionKind.Pass, frame.Source), "Keep this token in place"); break;
            case Prefix + "vip-drop": owner = frame.Target; Add(new(GameActionKind.PlaceToken, frame.Source), "Place VIP touching destroyed carrier", true); break;
            case Prefix + "fleet-ready":
                foreach (var token in game.State.Ships[frame.Source].GetDefenseTokenInstancesAll().Where(t => t.Exhausted))
                    Add(new(GameActionKind.ChooseToken, frame.Source, Token: token.Token, Index: token.Slot), "Discard objective token and ready physical defense token");
                Add(new(GameActionKind.Pass, frame.Source), "Keep objective token"); break;
            case Prefix + "asteroid-recover":
                foreach (var type in Enum.GetValues<DefenseTokenType>())
                {
                    int discarded = game.IsShip(frame.Source) ? game.State.Ships[frame.Source].GetDiscardedDefenseTokenCount(type) : game.State.Squadrons[game.SquadronIndex(frame.Source)].GetDiscardedDefenseTokenCount(type);
                    if (type != DefenseTokenType.Scatter && discarded > 0) Add(new(GameActionKind.ChooseToken, frame.Source, Token: type, Amount: 1), "Recover discarded " + type);
                    if (owner == Second(game))
                        foreach (var token in TokensOf(game, frame.Source, type).Where(t => t.Exhausted)) Add(new(GameActionKind.ChooseToken, frame.Source, Token: type, Index: token.Slot), "Ready " + type + " slot " + token.Slot);
                }
                Add(new(GameActionKind.Pass, frame.Source), "Decline asteroid recovery"); break;
            case Prefix + "planetary":
                foreach (var pair in PlanetaryTargets(game))
                    foreach (var zone in game.State.Ships[pair.ship].HullZones) Add(new(GameActionKind.Attack, frame.Source, pair.ship, ToZone: zone, Index: pair.token), "Remove objective token and attack chosen hull zone");
                Add(new(GameActionKind.Pass, frame.Source), "Decline Planetary Ion Cannon"); break;
            case Prefix + "hyperspace-reserve":
                if (frame.Stage == 0)
                    foreach (int ship in OfficialObjectives.LiveShips(game, owner).Where(s => game.Topology.Ships[s].CatalogEntry.Metadata.Size is "small" or "medium")) Add(new(GameActionKind.ChooseEntity, frame.Source, ship), "Set aside one small or medium ship");
                else if ((frame.Indices?.Length ?? 0) < 3)
                    foreach (var squad in game.Topology.Squadrons.Where(s => s.Owner == owner && !game.IsReserved(s.EntityIndex))) Add(new(GameActionKind.ChooseEntity, frame.Source, squad.EntityIndex), "Set aside squadron");
                if (frame.Stage > 0 || options.Count == 0) Add(new(GameActionKind.Pass, frame.Source), "Finish reserves"); break;
            case Prefix + "fighter-deploy":
                foreach (int squad in State(game).ReservedEntities.Where(e => !game.IsShip(e) && game.IsReserved(e))) Add(new(GameActionKind.Deploy, squad, Speed: 0), "Deploy ambushing squadron", true);
                break;
            case Prefix + "hyperspace-round":
                Add(new(GameActionKind.UseAbility, frame.Source), "Deploy all Hyperspace Assault reserves");
                if (MapMovementRules.Enabled(game))
                    foreach (int token in Tokens(game).Where(i => !(frame.Indices ?? []).Contains(i))) Add(new(GameActionKind.PlaceToken, frame.Source, token), "Move objective token up to distance 1", true);
                Add(new(GameActionKind.Pass, frame.Source), "Leave reserves set aside"); break;
            case Prefix + "hyperspace-token":
                foreach (int token in Tokens(game)) Add(new(GameActionKind.ChooseEntity, frame.Source, token), "Choose one entry token for all reserves"); break;
            case Prefix + "hyperspace-deploy":
                foreach (int entity in State(game).ReservedEntities.Where(game.IsReserved)) Add(new(GameActionKind.PlaceReservedUnit, entity, Speed: game.IsShip(entity) ? 1 : 0), "Deploy set-aside unit at objective token", true, game.IsShip(entity));
                break;
            case Prefix + "rift-setup":
                foreach (int ship in OfficialObjectives.LiveShips(game, First(game))) Add(new(GameActionKind.ChooseEntity, frame.Source, ship), "Choose enemy ship for forced straight speed-1 maneuver");
                Add(new(GameActionKind.Pass, frame.Source), "Decline Rift Ambush setup maneuver"); break;
            case Prefix + "rift-maneuver":
                options.AddRange(MovementModule.GetForcedManeuverActions(game, frame.Source, game.State.Owners[frame.Source] == Second(game) && frame.Text != "setup", ignoreGravityRift: frame.Text != "setup")); break;
            case Prefix + "rift-speed":
                owner = Second(game);
                for (int change = -1; change <= 1; change++)
                    if (game.State.Ships[frame.Source].CurrentSpeed + change >= 0 && game.State.Ships[frame.Source].CurrentSpeed + change <= game.Topology.Ships[frame.Source].CatalogEntry.Definition.MaximumSpeed)
                        Add(new(GameActionKind.ChangeSpeed, frame.Source, Speed: game.State.Ships[frame.Source].CurrentSpeed + change), "Set post-maneuver speed"); break;
            case Prefix + "outpost-score": case Prefix + "fire-lanes-score": case Prefix + "fleet-remove":
                return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "Resolve objective end-of-round effect")]);
            default: throw new NotSupportedException(frame.Rule);
        }
        if (options.Count == 0) return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "No remaining objective choices")]);
        return new(DecisionKind.Player, owner, GamePhase.AbilityChoice, options);
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!GetDecision(game, frame).Actions.Any(o => GameSession.Matches(o, action))) throw new ArgumentException("Illegal Defense objective action.");
        switch (frame.Rule)
        {
            case Prefix + "place-tokens":
                ValidateSetupToken(game, frame.Text, action.Position!.Value, frame.Stage);
                OfficialObjectives.PlaceToken(game, action.Position.Value, id: frame.Text);
                int total = frame.Text is "capture-the-vip" ? 1 : frame.Text is "jamming-barrier" ? 2 : 3;
                if (frame.Stage + 1 < total) { Replace(game, frame, frame with { Stage = frame.Stage + 1 }); return; } break;
            case Prefix + "fire-lanes-adjust":
                if (!MapMovementRules.Enabled(game)) break;
                if (action.Kind != GameActionKind.Pass) MoveToken(game, action.Target, action.Position!.Value, TabletopMeasurements.DistanceBand(2));
                if (frame.Stage < 2) { Replace(game, frame, frame with { Stage = frame.Stage + 1 }); return; } break;
            case Prefix + "vip-drop":
                if (!MapMovementRules.Enabled(game)) break;
                var circle = new Circle(action.Position!.Value, 12.7f); var carrier = game.ShipBase(frame.Source);
                if (!LegalVipDrop(game, carrier, circle)) throw new ArgumentException("VIP must touch the carrier's in-board base, or the closest play-area edge if the carrier is wholly outside.");
                OfficialObjectives.PlaceToken(game, circle.Center, id: "capture-the-vip"); break;
            case Prefix + "fleet-ready":
                if (action.Kind != GameActionKind.Pass)
                { if (!game.State.Ships[frame.Source].TryReadyDefenseToken(new DefenseTokenInstance(action.Token, action.Index, true))) throw new ArgumentException("Chosen token cannot be readied."); OfficialObjectives.SetTokens(game, frame.Source, OfficialObjectives.TokenCount(game, frame.Source) - 1); } break;
            case Prefix + "asteroid-recover": ResolveAsteroidRecovery(game, frame.Source, action); break;
            case Prefix + "planetary":
                Pop(game, frame); if (action.Kind == GameActionKind.Pass) return;
                OfficialObjectives.RemoveToken(game, action.Index); CombatModule.QueuePlanetaryAttack(game, action.Index, action.Target, action.ToZone, GamePhase.AbilityChoice); return;
            case Prefix + "hyperspace-reserve":
                if (action.Kind == GameActionKind.Pass) break;
                Reserve(game, action.Target);
                Replace(game, frame, frame.Stage == 0 ? frame with { Stage = 1, Indices = [] } : frame with { Indices = [..frame.Indices ?? [], action.Target] }); return;
            case Prefix + "fighter-deploy":
                if (action.Kind == GameActionKind.Pass) break;
                DeploySquadron(game, action.Source, action.Position!.Value, fighterAmbush: true);
                if (State(game).ReservedEntities.Any(game.IsReserved)) return; break;
            case Prefix + "hyperspace-round":
                if (action.Kind == GameActionKind.Pass) break;
                if (action.Kind == GameActionKind.PlaceToken)
                { MoveToken(game, action.Target, action.Position!.Value, TabletopMeasurements.DistanceOne); Replace(game, frame, frame with { Indices = [..frame.Indices ?? [], action.Target] }); return; }
                Replace(game, frame, frame with { Rule = Prefix + "hyperspace-token", Stage = 0 }); return;
            case Prefix + "hyperspace-token":
                State(game).Flags["hyperspace.entry"] = action.Target + 1;
                Replace(game, frame, frame with { Rule = Prefix + "hyperspace-deploy", Stage = 0 }); return;
            case Prefix + "hyperspace-deploy":
                if (action.Kind == GameActionKind.Pass) { foreach (int token in Tokens(game).ToArray()) OfficialObjectives.RemoveToken(game, token); break; }
                DeployHyperspace(game, action);
                if (!State(game).ReservedEntities.Any(game.IsReserved) && game.Flow.Phase != GamePhase.SquadronDisplacement)
                { foreach (int token in Tokens(game).ToArray()) OfficialObjectives.RemoveToken(game, token); Pop(game, frame); }
                return;
            case Prefix + "rift-setup":
                if (action.Kind == GameActionKind.Pass) break;
                Replace(game, frame, new(Prefix + "rift-maneuver", action.Target, Text: "setup")); return;
            case Prefix + "rift-maneuver":
                if (action.Kind == GameActionKind.Pass) break;
                bool allow = game.State.Owners[frame.Source] == Second(game) && frame.Text != "setup";
                if (frame.Text == "setup") Replace(game, frame, new(Prefix + "rift-speed", frame.Source)); else { State(game).Flags["rift.used." + frame.Source] = game.Flow.Flags["activation.serial"]; Pop(game, frame); }
                MovementModule.BeginForcedManeuver(game, frame.Source, action, allow, GamePhase.AbilityChoice, ignoreGravityRift: frame.Text != "setup"); return;
            case Prefix + "rift-speed": game.State.Ships[frame.Source].SetSpeed(action.Speed, game.Topology.Ships[frame.Source].CatalogEntry.Definition.MaximumSpeed); break;
            case Prefix + "outpost-score":
                foreach (int station in OfficialObjectives.Obstacles(game).Where(i => OfficialObjectives.ObstacleId(game, i) == "station")) AwardCommandMajority(game, station); break;
            case Prefix + "fire-lanes-score": foreach (int token in Tokens(game)) ResolveFireLane(game, token); break;
            case Prefix + "fleet-remove":
                foreach (int ship in OfficialObjectives.LiveShips(game))
                    if (OfficialObjectives.TokenCount(game, ship) > 0 && NearEdgeForFleet(game, ship)) OfficialObjectives.SetTokens(game, ship, OfficialObjectives.TokenCount(game, ship) - 1); break;
        }
        Pop(game, frame);
    }
    private static void Replace(GameContext game, GameContinuation frame, GameContinuation next) => game.Flow.Continuations[game.Flow.Continuations.IndexOf(frame)] = next;
    private static void Pop(GameContext game, GameContinuation frame) { game.Flow.Continuations.Remove(frame); game.Flow.Phase = GamePhase.AbilityChoice; }
    private static IEnumerable<int> Tokens(GameContext game) => OfficialObjectives.TokenIndices(game);
    private static IEnumerable<DefenseTokenInstance> TokensOf(GameContext game, int entity, DefenseTokenType type) => game.IsShip(entity)
        ? game.State.Ships[entity].GetDefenseTokenInstances(type) : game.State.Squadrons[game.SquadronIndex(entity)].GetDefenseTokenInstances(type);
    private static IEnumerable<(int token, int ship)> PlanetaryTargets(GameContext game)
    {
        foreach (int token in Tokens(game)) foreach (int ship in OfficialObjectives.LiveShips(game, First(game)))
            if (OfficialObjectives.DistanceToPiece(game, ship, token) <= TabletopMeasurements.DistanceBand(3) + Epsilon) yield return (token, ship);
    }
    private static bool AtRift(GameContext game, int ship) => OfficialObjectives.Obstacles(game).Any(i => OfficialObjectives.ObstacleId(game, i) == "gravity-rift" && OfficialObjectives.DistanceToPiece(game, ship, i) <= TabletopMeasurements.DistanceBand(2) + Epsilon);
    private static void Reserve(GameContext game, int entity)
    { game.State.Registers[game.Topology.Registers.EntityReserved(entity)] = 1; if (!State(game).ReservedEntities.Contains(entity)) State(game).ReservedEntities.Add(entity); }
    private static void ResolveAsteroidRecovery(GameContext game, int entity, GameAction action)
    {
        if (action.Kind == GameActionKind.Pass) return;
        if (action.Amount == 1)
        {
            var before = TokensOf(game, entity, action.Token).Select(t => t.Slot).ToArray();
            bool recovered = game.IsShip(entity) ? game.State.Ships[entity].TryRecoverDiscardedDefenseToken(action.Token) : game.State.Squadrons[game.SquadronIndex(entity)].TryRecoverDiscardedDefenseToken(action.Token);
            if (!recovered) throw new ArgumentException("No discarded defense token available.");
            if (game.State.Owners[entity] == First(game))
            {
                var token = TokensOf(game, entity, action.Token).Single(t => !before.Contains(t.Slot));
                if (game.IsShip(entity)) game.State.Ships[entity].TrySetDefenseTokenExhausted(token, true); else game.State.Squadrons[game.SquadronIndex(entity)].TrySetDefenseTokenExhausted(token, true);
            }
        }
        else if (game.IsShip(entity)) game.State.Ships[entity].TryReadyDefenseToken(new DefenseTokenInstance(action.Token, action.Index, true));
        else game.State.Squadrons[game.SquadronIndex(entity)].TryReadyDefenseToken(new DefenseTokenInstance(action.Token, action.Index, true));
    }
    private static void ValidateSetupToken(GameContext game, string id, Vector2 position, int stage)
    {
        Geometry2D.RequireFinite(position, nameof(position)); var circle = new Circle(position, 12.7f);
        var area = id is "hyperspace-assault" or "planetary-ion-cannon" ? MovementModule.Board(game) : OfficialObjectives.SetupArea(game);
        if (!PlayAreaGeometry.Contains(area, circle)) throw new ArgumentException("Objective token is outside its specified setup area.");
        float edge = id is "capture-the-vip" or "planetary-ion-cannon" ? TabletopMeasurements.DistanceBand(5) : id == "fire-lanes" ? TabletopMeasurements.DistanceBand(4) : id == "hyperspace-assault" ? TabletopMeasurements.DistanceBand(3) : 0;
        if (edge > 0 && (OfficialObjectives.DistanceFromPlayerEdge(game, position, 1) - circle.Radius <= edge || OfficialObjectives.DistanceFromPlayerEdge(game, position, 2) - circle.Radius <= edge)) throw new ArgumentException("Objective token is too close to a player edge.");
        if (id == "capture-the-vip")
        {
            if (!OfficialObjectives.Obstacles(game).Any(i => Geometry2D.Distance(circle, ObstacleGeometry.Shape(game, i)) <= TabletopMeasurements.DistanceOne + Epsilon)) throw new ArgumentException("VIP token must be at distance one of an obstacle.");
            var half = OfficialObjectives.SetupArea(game).HalfSize;
            if (half.X - MathF.Abs(position.X) - circle.Radius <= TabletopMeasurements.DistanceBand(5)) throw new ArgumentException("VIP token must be beyond distance five of every setup-area edge.");
        }
        if (id == "jamming-barrier" && stage > 0 && Geometry2D.Distance(circle, OfficialObjectives.PieceCircle(game, Tokens(game).Single())) > TabletopMeasurements.DistanceBand(5) + Epsilon)
            throw new ArgumentException("Jamming Barrier tokens must be within distance five.");
    }
    private static void MoveToken(GameContext game, int index, Vector2 position, float maximum)
    {
        var token = game.State.SpatialObjects[index]; var radius = MathF.Max(token.Size.X, token.Size.Y) / 2;
        Geometry2D.RequireFinite(position, nameof(position));
        if (Vector2.Distance(token.Pose.Position, position) > maximum + Epsilon || !PlayAreaGeometry.Contains(MovementModule.Board(game), new Circle(position, radius))) throw new ArgumentException("Illegal objective token movement.");
        TokenObstacleAttachments.MoveToken(game, index, new(position));
    }
    private static void DeploySquadron(GameContext game, int entity, Vector2 position, bool fighterAmbush)
    {
        if (!SquadronModule.IsLegalPlacement(game, entity, position)) throw new ArgumentException("Illegal reserved squadron position.");
        var circle = new Circle(position, game.SquadronBase(entity).Radius);
        if (fighterAmbush)
        {
            if (!PlayAreaGeometry.Contains(OfficialObjectives.SetupArea(game), circle) || OfficialObjectives.DistanceFromPlayerEdge(game, position, First(game)) - circle.Radius <= TabletopMeasurements.DistanceBand(5)) throw new ArgumentException("Fighter Ambush squadron is too close to the first player's edge.");
            bool normal = OfficialObjectives.LiveShips(game, Second(game)).Any(ship => Geometry2D.Distance(circle, game.ShipBase(ship)) <= TabletopMeasurements.DistanceBand(2));
            var definition = game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition;
            bool scout = CommandModule.KeywordValue(definition.Keywords, "SCOUT") > 0 || game.State.Registers[game.Topology.Registers[$"entity.{entity}.scoutGranted"]] != 0;
            if (scout && !Enumerable.Range(0, game.Topology.EntityCount).Any(other => OfficialObjectives.Live(game, other) && game.State.Owners[other] != game.State.Owners[entity] &&
                (game.IsShip(other) ? Geometry2D.Distance(circle, game.ShipBase(other)) : Geometry2D.Distance(circle, game.SquadronBase(other))) <= TabletopMeasurements.DistanceBand(5))) normal = true;
            bool obstacle = OfficialObjectives.Obstacles(game).Any(i => Geometry2D.Distance(circle, ObstacleGeometry.Shape(game, i)) <= TabletopMeasurements.DistanceOne);
            if (!normal && !obstacle) throw new ArgumentException("Fighter Ambush squadron must be near a friendly ship or obstacle.");
        }
        game.State.Poses[entity] = new(position); game.State.Registers[game.Topology.Registers.EntityReserved(entity)] = 0;
        game.State.Registers[game.Topology.Registers[$"entity.{entity}.placed"]] = 1;
    }
    private static void DeployHyperspace(GameContext game, GameAction action)
    {
        int entity = action.Source; if (!game.IsReserved(entity) || !State(game).ReservedEntities.Contains(entity) || action.Position is not { } position) throw new ArgumentException("Unit is not in Hyperspace Assault reserves.");
        var pose = new Pose(position, action.YawDegrees);
        if (!game.IsShip(entity))
        {
            int entry = (int)State(game).Flags["hyperspace.entry"] - 1;
            if (Geometry2D.Distance(new Circle(position, game.SquadronBase(entity).Radius), OfficialObjectives.PieceCircle(game, entry)) > TabletopMeasurements.DistanceOne) throw new ArgumentException("All reserves must deploy at the one chosen entry token.");
            DeploySquadron(game, entity, position, false); return;
        }
        var shape = new OrientedRectangle(pose, game.Topology.Ships[entity].FootprintMillimetres);
        int entryToken = (int)State(game).Flags["hyperspace.entry"] - 1;
        if (!PlayAreaGeometry.Contains(MovementModule.Board(game), shape) || Geometry2D.Distance(OfficialObjectives.PieceCircle(game, entryToken), shape) > TabletopMeasurements.DistanceOne ||
            OfficialObjectives.LiveShips(game).Any(s => s != entity && Geometry2D.Overlaps(game.ShipBase(s), shape))) throw new ArgumentException("Illegal Hyperspace Assault ship position.");
        int max = game.Topology.Ships[entity].CatalogEntry.Definition.ChartedMaximumSpeed;
        if (action.Speed < 0 || action.Speed > max) throw new ArgumentException("Illegal reserved ship speed.");
        game.State.Poses[entity] = pose; game.State.Ships[entity].SetSpeed(action.Speed, max);
        OfficialCreatureObjectives.ApplyDeploymentSpeed(game, entity);
        game.State.Registers[game.Topology.Registers.EntityReserved(entity)] = 0; game.State.Registers[game.Topology.Registers[$"entity.{entity}.placed"]] = 1;
        MovementModule.ProcessDisplacements(game, entity);
        if (game.Flow.PendingDisplacements.Length > 0)
        { game.Flow.Flags["objective.displacement.return"] = (int)GamePhase.AbilityChoice; game.Flow.Phase = GamePhase.SquadronDisplacement; }
    }
    private static bool NearEdgeForFleet(GameContext game, int ship)
    {
        var shape = game.ShipBase(ship); var half = game.Topology.BoardSizeMillimetres / 2;
        Span<Vector2> corners = stackalloc Vector2[4]; shape.CopyCorners(corners);
        foreach (var corner in corners) if (half.X - MathF.Abs(corner.X) <= TabletopMeasurements.DistanceBand(5) || half.Y - MathF.Abs(corner.Y) <= TabletopMeasurements.DistanceBand(3)) return true;
        return false;
    }
    private static void AwardCommandMajority(GameContext game, int obstacle)
    {
        int[] total = new int[2]; foreach (int ship in OfficialObjectives.LiveShips(game))
            if (OfficialObjectives.DistanceToPiece(game, ship, obstacle) <= TabletopMeasurements.DistanceOne) total[game.State.Owners[ship] - 1] += game.Topology.Ships[ship].CatalogEntry.Definition.Command;
        if (total[0] != total[1]) OfficialObjectives.AwardVictory(game, total[0] > total[1] ? 1 : 2);
    }
    private static void ResolveFireLane(GameContext game, int token)
    {
        bool[] overlaps = new bool[2]; for (int entity = 0; entity < game.Topology.EntityCount; entity++)
            if (OfficialObjectives.Live(game, entity) && OfficialObjectives.DistanceToPiece(game, entity, token) <= Epsilon) overlaps[game.State.Owners[entity] - 1] = true;
        if (overlaps[0] || overlaps[1]) { if (overlaps[0] != overlaps[1]) OfficialObjectives.AwardVictory(game, overlaps[0] ? 2 : 1); return; }
        int[] dice = new int[2]; var marker = OfficialObjectives.PieceCircle(game, token);
        foreach (int ship in OfficialObjectives.LiveShips(game)) foreach (var zone in game.State.Ships[ship].HullZones)
        {
            var geometry = game.ShipViews[ship].HullGeometryFor(zone);
            if (!CircleInArc(geometry, marker)) continue;
            var measure = HullGeometry.MeasureToSquadron(geometry, marker);
            var range = HullGeometry.ClassifyRange(measure.Distance);
            if (range == RangeBand.OutOfRange) continue;
            using var scope = new GameScope(game);
            var battery = UpgradeAttackDice.EffectiveBatteryArmament(game.ShipViews[ship], zone);
            int count = battery.RedDice + (range <= RangeBand.Medium ? battery.BlueDice : 0) + (range == RangeBand.Close ? battery.BlackDice : 0);
            if (count <= 0) continue;
            var line = new Segment(geometry.TargetingPoint, marker.Center);
            var blockers = new List<LineOfSightObstacle>();
            foreach (int other in OfficialObjectives.LiveShips(game).Where(s => s != ship)) blockers.Add(new(game.ShipBase(other)));
            foreach (int obstacle in OfficialObjectives.Obstacles(game))
                if (!(OfficialObjectives.Is(game, "contested-outpost") && OfficialObjectives.ObstacleId(game, obstacle) == "station")) blockers.Add(new(ObstacleGeometry.Shape(game, obstacle)));
            if (LineOfSightGeometry.Evaluate(line, blockers.ToArray()) == LineOfSightResult.Obstructed) count--;
            dice[game.State.Owners[ship] - 1] += Math.Max(0, count);
        }
        if (dice[0] != dice[1]) OfficialObjectives.AwardVictory(game, dice[0] > dice[1] ? 1 : 2);
    }
    public static int ExtraPreRollRemovals(GameContext game, AttackState attack)
    {
        if (!OfficialObjectives.Is(game, "jamming-barrier") || attack.Attacker < 0) return 0;
        int[] tokens = Tokens(game).ToArray(); if (tokens.Length != 2) return 0;
        if (tokens.Any(t => OfficialObjectives.DistanceToPiece(game, attack.Attacker, t) <= Epsilon || OfficialObjectives.DistanceToPiece(game, attack.Defender, t) <= Epsilon)) return 0;
        Vector2 start = game.IsShip(attack.Attacker) ? game.ShipViews[attack.Attacker].HullGeometryFor(attack.FromZone).TargetingPoint : game.State.Poses[attack.Attacker].Position;
        Vector2 end = game.IsShip(attack.Defender) ? game.ShipViews[attack.Defender].HullGeometryFor(attack.ToZone).TargetingPoint : game.State.Poses[attack.Defender].Position;
        var line = new Segment(start, end); var barrier = new Segment(game.State.SpatialObjects[tokens[0]].Pose.Position, game.State.SpatialObjects[tokens[1]].Pose.Position);
        bool crosses = Geometry2D.ClosestPoints(line, barrier).Distance <= Epsilon || tokens.Any(t => Vector2.Distance(game.State.SpatialObjects[t].Pose.Position, Geometry2D.ClosestPoint(line, game.State.SpatialObjects[t].Pose.Position)) <= 12.7f + Epsilon);
        return crosses ? attack.Dice.Length / 2 : 0;
    }
    private static bool LegalVipDrop(GameContext game, OrientedRectangle carrier, Circle token)
    {
        var board = MovementModule.Board(game); if (!PlayAreaGeometry.Contains(board, token)) return false;
        if (Geometry2D.Contact(carrier, board) != ContactKind.Separated)
            return MathF.Abs(Vector2.Distance(carrier.ClosestPoint(token.Center), token.Center) - token.Radius) <= Epsilon;
        Span<Vector2> boardCorners = stackalloc Vector2[4]; board.CopyCorners(boardCorners);
        Span<Vector2> shipCorners = stackalloc Vector2[4]; carrier.CopyCorners(shipCorners);
        Span<float> distances = stackalloc float[4]; float closest = float.PositiveInfinity;
        for (int edge = 0; edge < 4; edge++)
        {
            var segment = new Segment(boardCorners[edge], boardCorners[(edge + 1) % 4]);
            distances[edge] = float.PositiveInfinity;
            for (int side = 0; side < 4; side++) distances[edge] = MathF.Min(distances[edge],
                Geometry2D.ClosestPoints(segment, new Segment(shipCorners[side], shipCorners[(side + 1) % 4])).Distance);
            closest = MathF.Min(closest, distances[edge]);
        }
        for (int edge = 0; edge < 4; edge++)
            if (distances[edge] <= closest + Epsilon && MathF.Abs(Vector2.Distance(token.Center,
                    Geometry2D.ClosestPoint(new Segment(boardCorners[edge], boardCorners[(edge + 1) % 4]), token.Center)) - token.Radius) <= Epsilon) return true;
        return false;
    }
    private static bool CircleInArc(HullZoneGeometry arc, Circle target)
    {
        if (arc.ContainsInArc(target.Center) || Vector2.Distance(arc.ArcOrigin, target.Center) <= target.Radius) return true;
        var offset = target.Center - arc.ArcOrigin;
        return Vector2.Dot(offset, arc.FirstBoundary) >= 0 && MathF.Abs(Geometry2D.Cross(offset, arc.FirstBoundary)) <= target.Radius ||
            Vector2.Dot(offset, arc.SecondBoundary) >= 0 && MathF.Abs(Geometry2D.Cross(offset, arc.SecondBoundary)) <= target.Radius;
    }
}
