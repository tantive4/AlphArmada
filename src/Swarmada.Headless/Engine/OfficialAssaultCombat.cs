using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public sealed partial class SourceCombatEffects
{
    private static bool IsShipTarget(GameContext game, AttackState a) => game.IsShip(a.Defender) || a.EffectFlags.ContainsKey("objective.station");
    private static int AttackOwner(GameContext game, AttackState a) => a.Attacker >= 0 ? game.State.Owners[a.Attacker]
        : (int)a.EffectFlags.GetValueOrDefault("neutral.owner", OfficialObjectives.SecondPlayer(game));
    private static bool CanSpendObjectiveDie(GameContext game, AttackState a, AttackDie die)
        => die.Face != DieFace.Blank && (die.Face != DieFace.Accuracy || !game.IsShip(a.Attacker) || game.State.Ships[a.Attacker].CanSpendAccuracyIcons);
    private static IEnumerable<AbilityAction> AssaultCombatActions(GameContext game, AttackState a, CombatWindow window)
    {
        string id = OfficialObjectives.Id(game);
        if (window == CombatWindow.CriticalEffects && a.EffectFlags.ContainsKey("neutral.planetary")
            && MissingSquadronCombatEffects.HasCritical(a, AttackDieColor.Blue)
            && a.EffectFlags.GetValueOrDefault("critical.count") == 0 && !a.FirstHullCardFaceup
            && !game.Topology.Ships[a.Defender].IsStation && !OfficerCombatRules.PreventsCriticalEffects(game.ShipViews[a.Defender])
            && !(a.Contain && OfficerCombatRules.ContainPreventsAllCriticalEffects(game.ShipViews[a.Defender]))
            && Enum.GetValues<DefenseTokenType>().Any(type => game.State.Ships[a.Defender].GetReadyDefenseTokenCount(type) > 0))
            yield return Ability("objective.assault.planetary-critical", "planetary-ion-cannon", "Planetary Ion Cannon: defender chooses and exhausts a token");
        if (id.Length == 0 || a.EnvironmentalDamage || a.Attacker < 0 || a.MineSpatialIndex >= 0 && !a.EffectFlags.ContainsKey("objective.station")) yield break;
        if (window == CombatWindow.AttackEffects)
        {
            bool ship = game.IsShip(a.Attacker), targetShip = game.IsShip(a.Defender);
            if (ship && targetShip && id == "most-wanted" && OfficialObjectives.IsObjectiveShip(game, a.Defender) && !a.IsSalvoAttack && !Used(a, "objective.assault.most-wanted"))
                foreach (var color in a.Dice.Select(die => die.Color).Distinct())
                    yield return Ability("objective.assault.most-wanted." + (int)color, id, "Most Wanted: add one " + color + " die");
            if (ship && targetShip && id == "close-range-intel-scan" && game.State.Ships[a.Attacker].CanSpendAccuracyIcons)
            {
                int cost = AttackOwner(game, a) == OfficialObjectives.SecondPlayer(game) ? 1 : 2;
                if (Indices(a, die => die.Face == DieFace.Accuracy).Count() >= cost)
                    yield return Ability("objective.assault.intel-scan", id, "Spend " + cost + " accuracy dice for a victory token");
            }
            if (targetShip && id == "precision-strike" && (ship || SquadronAbilityEngine.HasBomber(game.SquadronViews[game.SquadronIndex(a.Attacker)]))
                && game.State.Ships[a.Defender].FacedownDamageCardCount > 0 && Indices(a, die => die.Face is DieFace.Hit or DieFace.DoubleHit or DieFace.HitCritical).Any())
                yield return Ability("objective.assault.precision-strike", id, "Spend a hit die and flip a random facedown damage card");
            if (ship && targetShip && id == "rift-assault")
            {
                if (OfficialObjectives.TokenCount(game, a.Defender) == 0 && OfficialObjectives.Obstacles(game).All(piece => OfficialObjectives.DistanceToPiece(game, a.Defender, piece) > TabletopMeasurements.DistanceOne)
                    && Indices(a, die => CanSpendObjectiveDie(game, a, die)).Any())
                    yield return Ability("objective.assault.rift-assign", id, "Spend an icon die to mark the defender");
                if (OfficialObjectives.TokenCount(game, a.Defender) > 0 && RiftModifiableDice(game, a).Any())
                    yield return Ability("objective.assault.rift-change", id, "Discard defender's objective token, change a die, and gain a victory token");
            }
            if (ship && targetShip && id == "targeting-beacons" && AttackOwner(game, a) == OfficialObjectives.SecondPlayer(game)
                && MissingSquadronCombatEffects.HasModifiableDice(a)
                && OfficialObjectives.TokenIndices(game).Any(token => OfficialObjectives.DistanceToPiece(game, a.Defender, token) <= TabletopMeasurements.DistanceBand(2)))
                yield return Ability("objective.assault.beacon-reroll", id, "Targeting Beacons: reroll up to two dice");
        }
        if (window == CombatWindow.CriticalEffects && game.IsShip(a.Attacker) && game.IsShip(a.Defender) && !game.Topology.Ships[a.Defender].IsStation && !a.IsSalvoAttack
            && game.State.Ships[a.Attacker].CanResolveCriticalEffects && MissingSquadronCombatEffects.HasCritical(a)
            && !OfficerCombatRules.PreventsCriticalEffects(game.ShipViews[a.Defender])
            && !(a.Contain && OfficerCombatRules.ContainPreventsAllCriticalEffects(game.ShipViews[a.Defender]))
            && a.EffectFlags.GetValueOrDefault("critical.count") + (a.FirstHullCardFaceup ? 1 : 0) < (game.HasUpgrade(a.Attacker, "fire-control-team") ? 2 : 1))
        {
            if (id == "ion-storm") yield return Ability("objective.assault.ion-critical", id, "Ion Storm critical effect");
            if (id == "marked-for-destruction" && a.Range is RangeBand.Close or RangeBand.Medium)
                yield return Ability("objective.assault.marked-critical", id, "Mark the defending ship and remove all other enemy objective tokens");
        }
    }
    private static bool BeginAssaultInitialRoll(GameContext game, AttackState a)
    {
        if (!OfficialObjectives.Is(game, "opening-salvo") || !a.EffectFlags.Remove("objective.opening.pending") || a.IsSalvoAttack) return false;
        a.EffectFlags["objective.opening.resume"] = 1;
        if (AttackOwner(game, a) == OfficialObjectives.FirstPlayer(game))
        { AddRolled(game, a, AttackDieColor.Red, 2); return true; }
        Begin(a, "objective.assault.opening-color", "objective.assault.opening-salvo", AttackOwner(game, a), [], 2, 2); return true;
    }
    private static bool TryApplyAssaultCombat(GameContext game, AttackState a, GameAction action)
    {
        string id = action.AbilityId; if (!id.StartsWith("objective.assault.", StringComparison.Ordinal)) return false;
        int owner = AttackOwner(game, a);
        if (id.StartsWith("objective.assault.most-wanted.", StringComparison.Ordinal))
        { Mark(a, "objective.assault.most-wanted"); return AddRolled(game, a, (AttackDieColor)SuffixInt(id), 1); }
        if (id == "objective.assault.beacon-reroll")
        { Begin(a, "reroll", id, owner, MissingSquadronCombatEffects.ModifiableDice(a), 1, 2); return true; }
        if (id == "objective.assault.intel-scan")
        { int cost = owner == OfficialObjectives.SecondPlayer(game) ? 1 : 2; Begin(a, "objective.assault.intel-dice", id, owner, Indices(a, die => die.Face == DieFace.Accuracy), cost, cost); return true; }
        if (id == "objective.assault.precision-strike")
        { Begin(a, "objective.assault.precision-die", id, owner, Indices(a, die => die.Face is DieFace.Hit or DieFace.DoubleHit or DieFace.HitCritical), 1, 1); return true; }
        if (id == "objective.assault.rift-assign")
        { Begin(a, "objective.assault.rift-spend", id, owner, Indices(a, die => CanSpendObjectiveDie(game, a, die)), 1, 1); return true; }
        if (id == "objective.assault.rift-change")
        { Begin(a, "objective.assault.rift-change-die", id, owner, RiftModifiableDice(game, a), 1, 1); return true; }
        if (id is "objective.assault.ion-critical" or "objective.assault.marked-critical" or "objective.assault.planetary-critical")
        {
            a.EffectFlags["critical.count"] = a.EffectFlags.GetValueOrDefault("critical.count") + 1; Mark(a, id);
            if (id == "objective.assault.planetary-critical")
            { Begin(a, "objective.assault.planetary-token", id, game.State.Owners[a.Defender], [], 1, 1); return true; }
            if (id == "objective.assault.marked-critical")
            {
                for (int ship = 0; ship < game.State.Ships.Length; ship++) if (game.State.Owners[ship] != owner) OfficialObjectives.SetTokens(game, ship, 0);
                OfficialObjectives.SetTokens(game, a.Defender, 1); return true;
            }
            if (OfficialObjectives.TokenCount(game, a.Defender) > 0) OfficialObjectives.AwardVictory(game, owner);
            if (game.State.Commands[a.Defender].TokenCount == 0) game.State.Ships[a.Defender].LoseShields(a.ToZone, 1);
            else Begin(a, "objective.assault.ion-command", id, owner, [], 1, 1);
            return true;
        }
        return false;
    }
    private static IEnumerable<GameActionOption> AssaultCombatChoices(GameContext game, AttackState a)
    {
        if (a.EffectRule == "objective.assault.opening-color")
        {
            foreach (var color in new[] { AttackDieColor.Red, AttackDieColor.Blue, AttackDieColor.Black })
                yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, Amount: (int)color, AbilityId: "objective.assault.opening-color"), "Opening Salvo: add " + color + " die " + (a.EffectIndices.Length + 1));
            yield break;
        }
        if (a.EffectRule == "objective.assault.ion-command")
        {
            foreach (var command in game.State.Commands[a.Defender].Tokens)
                yield return new(new(GameActionKind.ChooseCommand, a.Attacker, a.Defender, Command: command, AbilityId: "objective.assault.ion-command"), "Discard enemy " + command + " token");
            yield break;
        }
        if (a.EffectRule == "objective.assault.planetary-token")
        {
            foreach (var type in Enum.GetValues<DefenseTokenType>())
                foreach (var token in game.State.Ships[a.Defender].GetDefenseTokenInstances(type).Where(token => !token.Exhausted))
                    yield return new(new(GameActionKind.ChooseToken, a.Defender, Token: type, Index: token.Slot, AbilityId: "objective.assault.planetary-token"), "Exhaust " + type + " slot " + token.Slot);
            yield break;
        }
        if (a.EffectRule == "objective.assault.rift-face")
        {
            foreach (var face in RiftFaces(game, a, a.EffectValue))
                yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, Amount: (int)face, AbilityId: "objective.assault.rift-face"), "Set die to " + face);
            yield break;
        }
        foreach (int index in a.EffectEligibleIndices.Where(index => !a.EffectIndices.Contains(index)))
            yield return new(new(GameActionKind.ChooseDie, a.Attacker, a.Defender, Index: index, AbilityId: "objective.assault.choose-die"), "Spend or change die " + index);
    }
    private static bool ApplyAssaultCombatChoice(GameContext game, AttackState a, GameAction action)
    {
        string rule = a.EffectRule, id = a.EffectActionId; int owner = a.EffectOwner;
        if (rule == "objective.assault.opening-color")
        {
            a.EffectIndices = [..a.EffectIndices, action.Amount];
            if (a.EffectIndices.Length < 2) return true;
            var colors = a.EffectIndices.ToArray(); int start = a.Dice.Length; ClearChoice(a);
            foreach (int color in colors) AppendDice(a, (AttackDieColor)color, 1);
            Reroll(game, a, [start, start + 1]); return true;
        }
        if (rule == "objective.assault.intel-dice")
        {
            a.EffectIndices = [..a.EffectIndices, action.Index];
            if (a.EffectIndices.Length < a.EffectMaximum) return true;
            MissingSquadronCombatEffects.RemoveDice(a, a.EffectIndices); OfficialObjectives.AwardVictory(game, owner); Mark(a, id); ClearChoice(a); return true;
        }
        if (rule == "objective.assault.precision-die")
        {
            MissingSquadronCombatEffects.RemoveDice(a, action.Index); Mark(a, id);
            var cards = game.State.Ships[a.Defender].DamageCards;
            Begin(a, "objective.assault.precision-random", id, owner, Enumerable.Range(0, cards.Count).Where(index => !cards[index].FaceUp), 1, 1); return true;
        }
        if (rule == "objective.assault.rift-spend")
        { MissingSquadronCombatEffects.RemoveDice(a, action.Index); OfficialObjectives.SetTokens(game, a.Defender, 1); Mark(a, id); ClearChoice(a); return true; }
        if (rule == "objective.assault.rift-change-die")
        { a.EffectValue = action.Index; a.EffectRule = "objective.assault.rift-face"; return true; }
        if (rule == "objective.assault.rift-face")
        {
            a.Dice[a.EffectValue] = new(a.Dice[a.EffectValue].Color, (DieFace)action.Amount);
            OfficialObjectives.SetTokens(game, a.Defender, OfficialObjectives.TokenCount(game, a.Defender) - 1);
            OfficialObjectives.AwardVictory(game, AttackOwner(game, a)); Mark(a, id); ClearChoice(a); return true;
        }
        if (rule == "objective.assault.ion-command")
        { if (!game.State.Commands[a.Defender].TrySpendToken(action.Command)) return false; ClearChoice(a); return true; }
        if (rule == "objective.assault.planetary-token")
        {
            var token = game.State.Ships[a.Defender].GetDefenseTokenInstances(action.Token).Single(token => token.Slot == action.Index);
            if (!game.State.Ships[a.Defender].SpendDefenseToken(token)) return false; ClearChoice(a); return true;
        }
        return false;
    }
    internal static bool HasObjectiveChance(AttackState a) => a.EffectRule == "objective.assault.precision-random";
    internal static GameAction SampleObjectiveChance(AttackState a, ref DeterministicRandom random)
        => new(GameActionKind.UseAbility, a.Attacker, a.Defender, Index: a.EffectEligibleIndices[random.NextInt(a.EffectEligibleIndices.Length)], AbilityId: "objective.assault.precision-random");
    internal static void AdvanceObjectiveChance(AttackState a, ref DeterministicRandom random) => random.NextInt(a.EffectEligibleIndices.Length);
    internal static void ApplyObjectiveChance(GameContext game, AttackState a, GameAction action)
    {
        if (action.Kind != GameActionKind.UseAbility || action.Source != a.Attacker || action.Target != a.Defender
            || action.AbilityId != "objective.assault.precision-random" || !a.EffectEligibleIndices.Contains(action.Index)) throw new ArgumentException("Invalid Precision Strike random-card outcome.");
        if (!game.State.Ships[a.Defender].FlipFacedownDamageCardFaceupAt(action.Index, out var card)) throw new ArgumentException("Randomly selected facedown card is unavailable.");
        ClearChoice(a); CombatModule.ResolveNewFaceupCard(game, a.Defender, card, GamePhase.ModifyAttack);
    }
    private static IEnumerable<DieFace> RiftFaces(GameContext game, AttackState a, int index)
    {
        bool second = AttackOwner(game, a) == OfficialObjectives.SecondPlayer(game);
        return Faces(a.Dice[index].Color).Where(face => face != a.Dice[index].Face
            && (second ? face != DieFace.Blank : face is DieFace.Hit or DieFace.Accuracy));
    }
    private static IEnumerable<int> RiftModifiableDice(GameContext game, AttackState a)
        => MissingSquadronCombatEffects.ModifiableDice(a).Where(index => RiftFaces(game, a, index).Any());
}
