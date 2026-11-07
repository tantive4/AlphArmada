using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public sealed partial class CombatModule
{
    public const string ConcentrateFireStarted = "command.cf-started";
    public const string ConcentrateFireMode = "command.cf-mode";
    public const string BeginConcentrateFireAction = "command.concentrate-fire.begin";
    public const string FinishConcentrateFireAction = "command.concentrate-fire.finish";
    private static bool HasConcentrateFireGrant(GameContext game, AttackState attack) => game.IsShip(attack.Attacker) &&
        (game.Flow.ShipActivations[attack.Attacker].ConcentrateFireDie || game.Flow.ShipActivations[attack.Attacker].ConcentrateFireReroll);
    private static bool NeedsConcentrateFireEntry(GameContext game, AttackState attack) =>
        attack.MineSpatialIndex < 0 && game.HasUpgrade(attack.Attacker, "director-krennic-officer") &&
        attack.Range is RangeBand.Medium or RangeBand.Long && attack.Dice.Where((die, index) =>
            die.Color == AttackDieColor.Red && MissingSquadronCombatEffects.IsDieModifiable(attack, index)).Any();
    private static bool IsConcentrateFireUpgrade(GameAction action) =>
        action.AbilityId.StartsWith("attack.dice.director-krennic", StringComparison.Ordinal) ||
        action.AbilityId.StartsWith("remaining.attack-dice.clone-gunners.", StringComparison.Ordinal);

    private static void StartConcentrateFire(GameContext game, AttackState attack, CommandSpendMode? mode = null)
    {
        if (attack.EffectFlags.ContainsKey(ConcentrateFireStarted)) return;
        var activation = game.Flow.ShipActivations[attack.Attacker];
        var inferred = activation.ConcentrateFireRavagerChoice ? CommandSpendMode.Token :
            activation.ConcentrateFireDie && activation.ConcentrateFireReroll ? CommandSpendMode.DialAndToken :
            activation.ConcentrateFireDie ? CommandSpendMode.Dial : CommandSpendMode.Token;
        string key = "activation.cf-mode." + attack.Attacker;
        attack.EffectFlags[ConcentrateFireMode] = (int)(mode ?? (game.Flow.Flags.TryGetValue(key, out long stored) ? (CommandSpendMode)stored : inferred));
        attack.EffectFlags[ConcentrateFireStarted] = 1;
    }

    private static void AddConcentrateFireCoreOptions(GameContext game, AttackState attack, List<GameActionOption> options)
    {
        var activation = game.Flow.ShipActivations[attack.Attacker];
        if (activation.ConcentrateFireDie)
            foreach (var color in attack.Dice.Select(die => die.Color).Distinct())
                options.Add(Option(new(GameActionKind.UseAbility, attack.Attacker, attack.Defender,
                    Amount: (int)color, AbilityId: "command.concentrate-fire.add-die"), "Add " + color + " die"));
        if (activation.ConcentrateFireReroll)
            for (int i = 0; i < attack.Dice.Length; i++)
                options.Add(Option(new(GameActionKind.UseAbility, attack.Attacker, attack.Defender,
                    DiceIndices: [i], AbilityId: "command.concentrate-fire.reroll"), "Reroll die " + i));
    }

    private GameDecision ConcentrateFireDecision(GameContext game, AttackState attack)
    {
        var options = new List<GameActionOption>();
        AddConcentrateFireCoreOptions(game, attack, options);
        options.AddRange(effects.GetActions(game, attack, attack.Window).Where(option => IsConcentrateFireUpgrade(option.Template)));
        options.Add(Option(new(GameActionKind.Pass, attack.Attacker, attack.Defender, AbilityId: FinishConcentrateFireAction),
            "Finish Concentrate Fire command, declining any remaining optional effects"));
        bool forced = options.Count == 1;
        return new(forced ? DecisionKind.Forced : DecisionKind.Player, forced ? 0 : Owner(game, attack.Attacker), game.Flow.Phase, options);
    }

    private static void FinishConcentrateFire(GameContext game, AttackState attack)
    {
        var activation = game.Flow.ShipActivations[attack.Attacker];
        activation.ConcentrateFireDie = false; activation.ConcentrateFireReroll = false; activation.ConcentrateFireRavagerChoice = false;
        attack.EffectFlags.Remove(ConcentrateFireStarted);
        DravenAbilityEffects.CompleteDeferred(game, attack.Attacker, ShipCommand.ConcentrateFire, game.Flow.Phase);
    }
}
