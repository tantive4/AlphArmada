using Armada.Core;
using Armada.Rules;

namespace Swarmada.Headless.Engine;

public sealed partial class SourceCombatEffects
{
    private static IEnumerable<AbilityAction> KrennicActions(GameContext game, AttackState attack)
    {
        if (!game.IsShip(attack.Attacker) || attack.IsSalvoAttack || attack.MineSpatialIndex >= 0 ||
            !game.HasUpgrade(attack.Attacker, "director-krennic-officer") ||
            attack.Range is not (RangeBand.Medium or RangeBand.Long) ||
            !attack.EffectFlags.ContainsKey("command.cf-started") ||
            !attack.EffectFlags.TryGetValue("command.cf-mode", out long encoded) || encoded is < 0 or > 2 ||
            !Indices(attack, die => die.Color == AttackDieColor.Red).Any()) yield break;
        var mode = (CommandSpendMode)encoded;
        if (CommandSpendRules.UsesDial(mode))
            yield return Ability("attack.dice.director-krennic.dial", "director-krennic-officer", "Krennic: reroll any red dice");
        if (CommandSpendRules.UsesToken(mode))
            yield return Ability("attack.dice.director-krennic.token", "director-krennic-officer", "Krennic: reroll up to two red dice");
    }
}
