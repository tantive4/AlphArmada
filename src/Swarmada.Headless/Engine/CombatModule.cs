using System.Numerics;
using Armada.Core;
using Armada.Data;
using Armada.Rules;
using Swarmada.Headless.Effects;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Synchronous, snapshot-owned attack sequence. Dice outcomes and physical defense-token
/// slots are explicit actions; no policy-shaped action indices, Unity objects or coroutines occur.</summary>
public sealed partial class CombatModule : IChanceGameplayModule, IVerifiedDecisionModule
{
    private readonly ICombatEffectHooks effects;
    public CombatModule(ICombatEffectHooks? effects = null) => this.effects = effects ?? new SourceCombatEffects();
    private static readonly DieFace[] RedFaces = [DieFace.Blank, DieFace.Blank, DieFace.Hit, DieFace.Hit,
        DieFace.Critical, DieFace.Critical, DieFace.Accuracy, DieFace.DoubleHit];
    private static readonly DieFace[] BlueFaces = [DieFace.Hit, DieFace.Hit, DieFace.Hit, DieFace.Hit,
        DieFace.Critical, DieFace.Critical, DieFace.Accuracy, DieFace.Accuracy];
    private static readonly DieFace[] BlackFaces = [DieFace.Blank, DieFace.Blank, DieFace.Hit, DieFace.Hit,
        DieFace.Hit, DieFace.Hit, DieFace.HitCritical, DieFace.HitCritical];
    private static readonly HashSet<string> BuiltinUpgrades = new(StringComparer.Ordinal)
    {
        "xi7-turbolasers", "heavy-turbolaser-turrets", "sw-7-ion-batteries", "advanced-projectors",
        "foresight-title", "mon-mothma-commander", "damage-control-officer", "commander-gherant-officer",
        "bright-hope-title", "enhanced-armament", "spinal-armament", "expanded-launchers", "quad-laser-turrets",
        "general-dodonna-commander", "concord-title", "iden-versio-officer", "hardened-bulkheads",
        "medical-team", "lira-wessex-officer", "tide-of-progress-xii-title", "gallant-haven-title", "gunnery-team", "ravager-title", "proximity-mines"
    };
    public static bool SupportsUpgrade(string id) => BuiltinUpgrades.Contains(id)
        || new SourceCombatEffects().SupportsUpgrade(id) || EngineEffects.SupportsUpgrade(id);

    public bool Handles(GamePhase phase) => phase is GamePhase.SelectAttack or GamePhase.RollAttack
        or GamePhase.ModifyAttack or GamePhase.SpendAccuracy or GamePhase.SpendDefense or GamePhase.EvadeDice
        or GamePhase.RedirectDamage or GamePhase.ResolveCritical or GamePhase.DealDamage or GamePhase.ChooseDamageCard;

    public GameDecision GetDecision(GameContext game)
    {
        using var scope = new GameScope(game);
        if (game.Flow.Phase == GamePhase.SelectAttack) return AttackSelection(game);
        var a = Current(game);
        if (!a.EnvironmentalDamage && !a.Rolled && a.Dice.Length == 0)
            return new(DecisionKind.Forced, 0, game.Flow.Phase, [Option(new(GameActionKind.Pass, a.Attacker, a.Defender, AbilityId: "core.cancel-empty"), "Cancel attack with an empty dice pool")]);
        if (SourceCombatEffects.HasObjectiveChance(a)) return new(DecisionKind.Chance, 0, game.Flow.Phase, []);
        if (a.EffectFlags.ContainsKey("objective.solar-pending"))
            return new(DecisionKind.Player, Owner(game, a.Attacker), game.Flow.Phase, OfficialNavigationObjectives.SolarCoronaOptions(game, a).ToArray());
        if (a.EffectFlags.ContainsKey("completion.resume"))
            return new(DecisionKind.Forced, 0, game.Flow.Phase, [Option(new(GameActionKind.Pass, a.Attacker, a.Defender), "Complete defending attack")]);
        var options = new List<GameActionOption>();
        int owner = a.EffectFlags.ContainsKey("neutral.ship") ? (int)a.EffectFlags["neutral.owner"] : Owner(game, a.Attacker >= 0 ? a.Attacker : a.Defender);
        DecisionKind kind = DecisionKind.Player;
        if (effects.HasPendingChoice(game, a))
            return new(DecisionKind.Player, effects.DecisionPlayer(game, a) ?? owner, game.Flow.Phase,
                effects.GetActions(game, a, a.Window).ToArray());
        if (game.Flow.Phase == GamePhase.ModifyAttack && a.Rolled && a.EffectFlags.ContainsKey(ConcentrateFireStarted))
            return ConcentrateFireDecision(game, a);
        switch (game.Flow.Phase)
        {
            case GamePhase.RollAttack:
                return new(DecisionKind.Chance, 0, game.Flow.Phase, []);
            case GamePhase.ModifyAttack:
                if (a.Rolled && a.EffectFlags.ContainsKey("attack-effects.finished"))
                { kind = DecisionKind.Forced; options.Add(Option(new(GameActionKind.Pass, a.Attacker, a.Defender), "Continue after attack effects")); break; }
                if (!a.Rolled && a.MandatoryRemovals > 0 && a.EffectFlags.GetValueOrDefault("preroll.jamming.remaining") == 0
                    && game.IsShip(a.Defender) && game.HasUpgrade(a.Defender, "thermal-shields") && !a.EffectFlags.ContainsKey("thermal-window.offered"))
                {
                    kind = DecisionKind.Forced;
                    options.Add(Option(new(GameActionKind.Pass, a.Attacker, a.Defender, AbilityId: "core.thermal-before-obstruction"), "Resolve Thermal Shields before obstruction"));
                }
                else if (!a.Rolled && a.MandatoryRemovals > 0)
                {
                    for (int i = 0; i < a.Dice.Length; i++)
                        options.Add(Option(new(GameActionKind.ModifyDice, a.Attacker, a.Defender,
                            Index: i, AbilityId: "core.remove-gathered-die"), "Remove " + a.Dice[i].Color + " die"));
                }
                else
                {
                    AddEffectOptions(game, a, options);
                    if (a.Rolled && game.IsShip(a.Attacker) && !a.IsSalvoAttack)
                    {
                        var activation = game.Flow.ShipActivations[a.Attacker];
                        AddConcentrateFireCoreOptions(game, a, options);
                        if (HasConcentrateFireGrant(game, a))
                        {
                            options.Add(Option(new(GameActionKind.Pass, a.Attacker, a.Defender, AbilityId: FinishConcentrateFireAction),
                                "Resolve Concentrate Fire without changing dice"));
                            if (NeedsConcentrateFireEntry(game, a))
                                options.Add(Option(new(GameActionKind.UseAbility, a.Attacker, a.Defender, AbilityId: BeginConcentrateFireAction),
                                    "Resolve Concentrate Fire with optional Krennic effects"));
                        }
                        if (ActivationAbilityEffects.CanUseBeck(game, ShipCommand.ConcentrateFire, true))
                            for (int i = 0; i < a.Dice.Length; i++)
                                options.Add(Option(new(GameActionKind.UseAbility, a.Attacker, a.Defender, DiceIndices: [i],
                                    AbilityId: "combat.beck.cf-reroll"), "Beck: gain CF raid and reroll die " + i));
                        var commandTokens = CommandModule.AdditionalTokenOptions(game, GamePhase.ModifyAttack);
                        options.AddRange(commandTokens.Where(option => option.Template.Command == ShipCommand.ConcentrateFire));
                        if (commandTokens.Any(option => option.Template.Command == ShipCommand.ConcentrateFire))
                            foreach (int upgrade in CommandModule.TokenStrengthUpgradeIndices(game, ShipCommand.ConcentrateFire))
                                options.Add(Option(new(GameActionKind.UseAbility, a.Attacker, Command: ShipCommand.ConcentrateFire,
                                    SpendMode: CommandSpendMode.Token, Index: upgrade, AbilityId: "command.resolve-strengthened-token"),
                                    "Resolve CF token at dial strength with " + game.State.Upgrades[upgrade].Definition.name));
                    }
                    options.Add(Option(new(GameActionKind.Pass, a.Attacker, a.Defender), a.Rolled ? "Finish attack effects" : "Roll attack dice"));
                }
                break;
            case GamePhase.SpendAccuracy:
                if (!game.IsShip(a.Attacker) || game.State.Ships[a.Attacker].CanSpendAccuracyIcons)
                    for (int die = 0; die < a.Dice.Length; die++)
                        if (a.Dice[die].Face == DieFace.Accuracy && MissingSquadronCombatEffects.IsDieModifiable(a, die))
                            foreach (var token in Tokens(game, a.Defender).Where(token => !Locked(a, token)))
                                options.Add(Option(new(GameActionKind.SpendAccuracy, a.Attacker, a.Defender,
                                    Token: token.Token, Index: token.Slot, DiceIndices: [die]), $"Lock {token.Token} token {token.Slot}"));
                options.Add(Option(new(GameActionKind.FinishAccuracy, a.Attacker, a.Defender), "Finish accuracy"));
                break;
            case GamePhase.SpendDefense:
                owner = Owner(game, a.Defender);
                if (a.EffectFlags.ContainsKey("defense.finished"))
                { kind = DecisionKind.Forced; options.Add(Option(new(GameActionKind.FinishDefense, a.Defender, a.Attacker), "Resolve damage after defense timing")); break; }
                foreach (var token in Tokens(game, a.Defender).Where(token => CanSpend(game, a, token)))
                    options.Add(Option(new(GameActionKind.SpendDefenseToken, a.Defender, a.Attacker,
                        Token: token.Token, Index: token.Slot), $"Spend {token.Token} token {token.Slot}"));
                AddEffectOptions(game, a, options);
                options.Add(Option(new(GameActionKind.FinishDefense, a.Defender, a.Attacker), "Resolve incoming damage"));
                break;
            case GamePhase.EvadeDice:
                owner = Owner(game, a.Defender);
                int maximum = Math.Min(a.Dice.Length, a.EvadeBaseCount + (CanDiscardExtraEvade(game, a) ? 1 : 0));
                foreach (var indices in Subsets(a.Dice.Length, maximum).Where(indices => indices.All(index => MissingSquadronCombatEffects.IsDieModifiable(a, index))))
                    options.Add(Option(new(GameActionKind.ChooseEvadeDice, a.Defender, a.Attacker, DiceIndices: indices),
                        (a.EvadeCancels ? "Cancel " : "Reroll ") + string.Join(",", indices)));
                break;
            case GamePhase.RedirectDamage:
                owner = Owner(game, a.Defender);
                var allocations = a.RedirectAllocations.ToDictionary(x => x.Key, x => x.Value);
                int remaining = a.Damage - allocations.Values.Sum();
                int zones = DefenseUpgradeEffects.MaximumRedirectZones(game.HasUpgrade(a.Defender, "advanced-projectors"), game.HasUpgrade(a.Defender, "foresight-title"));
                foreach (var zone in DefenseResolver.GetRedirectCandidates(a.ToZone,
                    game.HasUpgrade(a.Defender, "advanced-projectors"), game.State.Ships[a.Defender].HasAuxiliaryHullZones))
                {
                    int allocated = allocations.GetValueOrDefault(zone);
                    if (allocated == 0 && allocations.Count >= zones) continue;
                    int cap = game.HasUpgrade(a.Attacker, "xi7-turbolasers") ? 1 : int.MaxValue;
                    int count = Math.Min(remaining, Math.Min(game.State.Ships[a.Defender].GetShields(zone) - allocated, cap - allocated));
                    for (int amount = 1; amount <= count; amount++)
                        options.Add(Option(new(GameActionKind.Redirect, a.Defender, a.Attacker, ToZone: zone, Amount: amount), $"Redirect {amount} to {zone}"));
                }
                options.Add(Option(new(GameActionKind.FinishRedirect, a.Defender, a.Attacker), "Finish redirect allocation"));
                break;
            case GamePhase.ResolveCritical:
                if (a.PendingDamageCard.HasValue) return DamageCardDecision(game, a);
                AddEffectOptions(game, a, options);
                if (!a.IsSalvoAttack && !Station(game, a.Defender) && effects.AllowsStandardCritical(game, a)) options.AddRange(OfficialNavigationObjectives.CriticalOptions(game, a));
                if (CanStandardCritical(game, a) && effects.AllowsStandardCritical(game, a))
                    options.Add(Option(new(GameActionKind.ChooseCritical, a.Attacker, a.Defender, AbilityId: "critical.standard"), "First hull damage card faceup"));
                options.Add(Option(new(GameActionKind.ChooseCritical, a.Attacker, a.Defender, AbilityId: "critical.none"), "Finish critical effects"));
                break;
            case GamePhase.DealDamage:
                owner = Owner(game, a.Defender);
                if (a.NeedsDamageZone && game.IsShip(a.Defender))
                    foreach (var zone in TargetZones(game, a.Defender))
                        options.Add(Option(new(GameActionKind.ChooseZone, Target: a.Defender, ToZone: zone), "Suffer damage in " + zone));
                else
                {
                    kind = DecisionKind.Forced;
                    options.Add(Option(new(GameActionKind.ResolveDamage, a.Attacker, a.Defender), "Apply damage"));
                }
                break;
            case GamePhase.ChooseDamageCard:
                if (a.IncomingDamageCard.HasValue)
                    return new(DecisionKind.Player, Owner(game, a.Defender), game.Flow.Phase, IncomingCardOptions(game, a).ToArray());
                int look = Dodonna(game, a) ? Math.Min(4, Deck(game, a.Defender).Remaining) : 1;
                if (look > 1)
                {
                    owner = Owner(game, a.Attacker);
                    var cards = Deck(game, a.Defender).PeekTop(look);
                    for (int i = 0; i < cards.Count; i++)
                        options.Add(Option(new(GameActionKind.DrawDamageCard, a.Attacker, a.Defender, Index: i), "Deal " + cards[i]));
                }
                else
                {
                    kind = DecisionKind.Forced;
                    options.Add(Option(new(GameActionKind.DrawDamageCard, a.Attacker, a.Defender, Index: 0), "Draw damage card"));
                }
                break;
            default: throw new InvalidOperationException("Combat does not handle " + game.Flow.Phase);
        }
        return new(kind, kind == DecisionKind.Forced ? 0 : owner, game.Flow.Phase, options);
    }

    public GameAction SampleChance(GameContext game, ref DeterministicRandom random)
    {
        if (game.Flow.Attacks.Count > 0 && SourceCombatEffects.HasObjectiveChance(Current(game))) return SourceCombatEffects.SampleObjectiveChance(Current(game), ref random);
        if (game.Flow.Phase != GamePhase.RollAttack) throw new InvalidOperationException("No pending combat chance event.");
        var a = Current(game); var dice = a.Dice.ToArray();
        foreach (int index in a.PendingRollIndices)
        {
            var faces = Faces(dice[index].Color);
            dice[index] = new(dice[index].Color, faces[random.NextInt(faces.Length)]);
        }
        return new(GameActionKind.ResolveDice, a.Attacker, a.Defender, Dice: dice);
    }

    public void AdvanceChanceRandom(GameContext game, ref DeterministicRandom random)
    {
        var attack = Current(game);
        if (SourceCombatEffects.HasObjectiveChance(attack)) { SourceCombatEffects.AdvanceObjectiveChance(attack, ref random); return; }
        foreach (int index in attack.PendingRollIndices) random.NextInt(Faces(attack.Dice[index].Color).Length);
    }

    public void Apply(GameContext game, GameAction action) => ApplyCore(game, action, verified: false);
    void IVerifiedDecisionModule.ApplyVerified(GameContext game, GameAction action) => ApplyCore(game, action, verified: true);
    private void ApplyCore(GameContext game, GameAction action, bool verified)
    {
        using var scope = new GameScope(game);
        if (game.Flow.Phase == GamePhase.SelectAttack)
        {
            if (!verified) RequireOffered(game, action);
            if (CommandModule.ResolveActivationCommand(game, action, GamePhase.SelectAttack)) return;
            if (action.Kind == GameActionKind.EndActivation) { RoundModule.EndShipActivation(game); return; }
            if (action.Kind == GameActionKind.SkipAttack)
            {
                if (game.Flow.Flags.ContainsKey("combat.granted.active"))
                { game.Flow.Flags.Remove(FlakKey(game, action.Source)); game.Flow.Phase = (GamePhase)game.Flow.Flags["combat.granted.return"]; ClearGrantedFlags(game); return; }
                int active = game.Flow.ActiveShip;
                if (game.Flow.Flags.ContainsKey($"combat.demolisher.after-maneuver.{game.State.Round}.{active}"))
                { game.Flow.Flags.Remove(FlakKey(game, active)); game.Flow.Phase = GamePhase.AbilityChoice; return; }
                if (game.Flow.ActiveSquadron >= 0) { game.Flow.SquadronCanAttack = false; game.Flow.Phase = GamePhase.SquadronActivation; return; }
                if (FlakZone(game, active).HasValue) { game.Flow.Flags.Remove(FlakKey(game, active)); return; }
                game.Flow.Phase = GamePhase.SelectCourse; return;
            }
            BeginAttack(game, action); return;
        }
        var a = Current(game);
        if (action.AbilityId == "core.cancel-empty")
        { if (!verified) RequireOffered(game, action); CompleteAttack(game, a); return; }
        if (SourceCombatEffects.HasObjectiveChance(a)) { SourceCombatEffects.ApplyObjectiveChance(game, a, action); return; }
        if (a.EffectFlags.ContainsKey("objective.solar-pending"))
        {
            if (!verified) RequireOffered(game, action);
            if (!OfficialNavigationObjectives.ApplySolarCorona(game, a, action)) throw new ArgumentException("Illegal Solar Corona cancellation.");
            a.EffectFlags.Remove("objective.solar-pending"); effects.EnterWindow(game, a, CombatWindow.AttackEffects); return;
        }
        if (a.EffectFlags.ContainsKey("completion.resume"))
        { if (!verified) RequireOffered(game, action); a.EffectFlags.Remove("completion.resume"); CompleteAttack(game, a); return; }
        if (game.Flow.Phase == GamePhase.RollAttack) { ApplyRoll(game, a, action); return; }
        if (!verified) RequireOffered(game, action);
        if (action.AbilityId == BeginConcentrateFireAction) { StartConcentrateFire(game, a); return; }
        if (action.AbilityId == FinishConcentrateFireAction) { StartConcentrateFire(game, a); FinishConcentrateFire(game, a); return; }
        if (action.AbilityId == "core.thermal-before-obstruction")
        {
            if (!effects.TryFinishWindow(game, a, CombatWindow.BeforeRoll)) a.EffectFlags["thermal-window.offered"] = 1;
            return;
        }
        if (effects.HasPendingChoice(game, a))
        {
            if (!effects.TryApply(game, a, a.Window, action)) throw new ArgumentException("Invalid pending combat effect choice.", nameof(action));
            if (!effects.HasPendingChoice(game, a) && a.AfterAttackWindowEntered && !a.ResponseStarted
                && ReferenceEquals(game.Flow.Attacks.LastOrDefault(), a) && game.Flow.Phase != GamePhase.AbilityChoice) CompleteAttack(game, a);
            return;
        }
        if (game.Flow.Phase == GamePhase.ChooseDamageCard && a.IncomingDamageCard.HasValue)
        { ApplyIncomingCardChoice(game, a, action); return; }
        if (action.Kind == GameActionKind.UseAbility)
        {
            if (action.AbilityId == "command.resolve-additional-token")
            { CommandModule.ResolveAdditionalToken(game, action, game.Flow.Phase); return; }
            if (action.AbilityId == "command.resolve-strengthened-token")
            { CommandModule.ResolveStrengthenedToken(game, action, game.Flow.Phase); return; }
            if (action.AbilityId == "combat.beck.cf-reroll")
            {
                ActivationAbilityEffects.CommitBeckCommand(game, ShipCommand.ConcentrateFire);
                var activation = game.Flow.ShipActivations[a.Attacker];
                activation.ResolvedCommands |= 1 << (int)ShipCommand.ConcentrateFire; activation.ConcentrateFireReroll = false;
                StartConcentrateFire(game, a, CommandSpendMode.Token);
                DravenAbilityEffects.ArmDeferred(game, a.Attacker, ShipCommand.ConcentrateFire);
                game.Flow.Flags[$"round.resolved-command.{a.Attacker}"] = game.State.Round;
                a.PendingRollIndices = action.DiceIndices!.ToArray(); a.AfterRollPhase = GamePhase.ModifyAttack; game.Flow.Phase = GamePhase.RollAttack; return;
            }
            if (action.AbilityId is "command.concentrate-fire.add-die" or "command.concentrate-fire.reroll")
            {
                StartConcentrateFire(game, a);
                var activation = game.Flow.ShipActivations[a.Attacker];
                if (action.AbilityId.EndsWith("add-die", StringComparison.Ordinal))
                {
                    activation.ConcentrateFireDie = false;
                    a.Dice = [..a.Dice, new AttackDie((AttackDieColor)action.Amount, DieFace.Blank)];
                    a.PendingRollIndices = [a.Dice.Length - 1];
                }
                else { activation.ConcentrateFireReroll = false; a.PendingRollIndices = action.DiceIndices!.ToArray(); }
                if (activation.ConcentrateFireRavagerChoice)
                { activation.ConcentrateFireDie = false; activation.ConcentrateFireReroll = false; activation.ConcentrateFireRavagerChoice = false; }
                a.AfterRollPhase = GamePhase.ModifyAttack; game.Flow.Phase = GamePhase.RollAttack; return;
            }
            if (IsConcentrateFireUpgrade(action)) StartConcentrateFire(game, a);
            if (!effects.TryApply(game, a, a.Window, action))
                throw new NotSupportedException("Unimplemented combat effect action: " + action.AbilityId);
            return;
        }
        switch (game.Flow.Phase)
        {
            case GamePhase.ModifyAttack:
                if (action.Kind == GameActionKind.ModifyDice)
                {
                    MissingSquadronCombatEffects.RemoveDice(a, action.Index);
                    a.MandatoryRemovals--;
                    if (a.EffectFlags.GetValueOrDefault("preroll.jamming.remaining") > 0) a.EffectFlags["preroll.jamming.remaining"]--;
                    if (a.Dice.Length == 0) CompleteAttack(game, a);
                }
                else if (!a.Rolled)
                {
                    if (effects.TryFinishWindow(game, a, CombatWindow.BeforeRoll)) return;
                    ApplyAutomaticAttackBonuses(game, a);
                    a.PreRollComplete = true; a.PendingRollIndices = Enumerable.Range(0, a.Dice.Length).ToArray();
                    a.AfterRollPhase = GamePhase.ModifyAttack; game.Flow.Phase = GamePhase.RollAttack;
                }
                else
                {
                    a.EffectFlags["attack-effects.finished"] = 1;
                    if (!effects.TryFinishWindow(game, a, CombatWindow.AttackEffects))
                    { if (a.MineSpatialIndex >= 0) CompleteMineAttack(game, a); else game.Flow.Phase = GamePhase.SpendAccuracy; }
                }
                break;
            case GamePhase.SpendAccuracy:
                if (action.Kind == GameActionKind.FinishAccuracy) Enter(game, a, GamePhase.SpendDefense, CombatWindow.DefenseEffects);
                else
                {
                    var token = Tokens(game, a.Defender).Single(x => x.Token == action.Token && x.Slot == action.Index);
                    a.LockedTokens = [..a.LockedTokens, token];
                    MissingSquadronCombatEffects.RemoveDice(a, action.DiceIndices![0]);
                }
                break;
            case GamePhase.SpendDefense:
                if (action.Kind == GameActionKind.FinishDefense) FinishDefense(game, a);
                else SpendToken(game, a, action);
                break;
            case GamePhase.EvadeDice:
                ApplyEvade(game, a, action); break;
            case GamePhase.RedirectDamage:
                if (action.Kind == GameActionKind.FinishRedirect)
                {
                    a.Damage -= a.RedirectAllocations.Sum(x => x.Value);
                    Enter(game, a, GamePhase.ResolveCritical, CombatWindow.CriticalEffects);
                }
                else
                {
                    var requested = a.RedirectAllocations.ToDictionary(x => x.Key, x => x.Value);
                    requested[action.ToZone] = requested.GetValueOrDefault(action.ToZone) + action.Amount;
                    a.RedirectAllocations = requested.ToArray();
                }
                break;
            case GamePhase.ResolveCritical:
                if (a.PendingDamageCard.HasValue) ApplyDamageCardChoice(game, a, action);
                else
                {
                    if (OfficialNavigationObjectives.ApplyCritical(game, a, action))
                    { a.EffectFlags["critical.count"] = a.EffectFlags.GetValueOrDefault("critical.count") + 1; return; }
                    if (action.AbilityId == "critical.standard")
                    { a.FirstHullCardFaceup = true; return; }
                    a.CriticalResolved = true;
                    Enter(game, a, GamePhase.DealDamage, CombatWindow.BeforeDamage);
                }
                break;
            case GamePhase.DealDamage:
                if (action.Kind == GameActionKind.ChooseZone) { a.ToZone = action.ToZone; a.NeedsDamageZone = false; }
                else DealDamage(game, a);
                break;
            case GamePhase.ChooseDamageCard:
                if (a.IncomingDamageCard.HasValue) ApplyIncomingCardChoice(game, a, action);
                else DrawCard(game, a, action.Index);
                break;
            default: throw new InvalidOperationException("Invalid combat phase.");
        }
    }

    public static void QueueDamage(GameContext game, int defender, int amount, bool faceup, bool ignoreShields,
        GamePhase returnPhase, HullZone? zone = null, string rule = "environment.damage", int source = -1)
    {
        if (defender < 0 || defender >= game.Topology.EntityCount || amount < 0) throw new ArgumentOutOfRangeException(nameof(defender));
        var resume = game.Flow.Attacks.LastOrDefault() is { EnvironmentalDamage: true, DamageStarted: false }
            ? GamePhase.DealDamage : returnPhase;
        var a = new AttackState { Attacker = source, Defender = defender, Damage = amount,
            EnvironmentalDamage = true, IgnoreShields = ignoreShields, NeedsDamageZone = !ignoreShields && zone is null,
            ToZone = zone ?? HullZone.Front, FirstHullCardFaceup = faceup, ResumePhase = resume, DamageRule = rule };
        game.Flow.Attacks.Add(a); game.Flow.Phase = GamePhase.DealDamage;
    }

    /// <summary>Resolves the entry effect of a card already made faceup by another rule.</summary>
    public static void ResolveNewFaceupCard(GameContext game, int defender, CriticalDamageCard card, GamePhase returnPhase)
    {
        if (Station(game, defender))
        { game.State.Ships[defender].FlipCriticalFacedown(card); game.Flow.Phase = returnPhase; return; }
        OfficialAssaultObjectives.ObserveFaceupCard(game, defender);
        if (!game.IsShip(defender)) throw new ArgumentException("Damage cards belong to ships.", nameof(defender));
        if (card == CriticalDamageCard.LifeSupportFailure)
            foreach (var command in Enum.GetValues<ShipCommand>()) while (game.State.Commands[defender].TrySpendToken(command)) { }
        if (!CriticalEffectRegistry.Get(card).RequiresImmediatePrompt || game.IsDestroyed(defender))
        { game.Flow.Phase = returnPhase; return; }
        game.Flow.Attacks.Add(new AttackState { Attacker = -1, Defender = defender, EnvironmentalDamage = true,
            DamageStarted = true, PendingDamageCard = card, ResumePhase = returnPhase, DamageRule = "critical.entry." + card });
        game.Flow.Phase = GamePhase.ResolveCritical;
    }

    public static void QueueMineExplosion(GameContext game, int defender, int spatialIndex, GamePhase returnPhase)
    {
        if (!game.IsShip(defender) || spatialIndex < 0 || spatialIndex >= game.State.SpatialObjects.Count)
            throw new ArgumentOutOfRangeException(nameof(spatialIndex));
        var mine = game.State.SpatialObjects[spatialIndex];
        if (!mine.Active || mine.Kind != SetupModule.ProximityMineKind) throw new ArgumentException("Mine is unavailable.", nameof(spatialIndex));
        game.State.SpatialObjects[spatialIndex] = mine with { Active = false };
        TokenObstacleAttachments.ForgetToken(game, spatialIndex);
        game.Flow.Attacks.Add(new AttackState { Attacker = mine.SourceEntity, Defender = defender,
            EnvironmentalDamage = true, IgnoreShields = true, Dice = Unrolled(new(0, 2, 0)), PendingRollIndices = [0, 1],
            ResumePhase = returnPhase, DamageRule = "mine.explosion." + spatialIndex });
        game.Flow.Phase = GamePhase.RollAttack;
    }

    public static void QueueUnmodifiedDiceDamage(GameContext game, int source, int defender, int blueDice, GamePhase returnPhase, string rule)
    {
        var battery = new DiceBattery(0, blueDice, 0);
        game.Flow.Attacks.Add(new AttackState { Attacker = source, Defender = defender, EnvironmentalDamage = true,
            IgnoreShields = true, Dice = Unrolled(battery), PendingRollIndices = Enumerable.Range(0, blueDice).ToArray(),
            ResumePhase = returnPhase, DamageRule = rule }); game.Flow.Phase = GamePhase.RollAttack;
    }

    /// <summary>Enters a rule-granted attack without consuming an ordinary squadron activation.
    /// The granting rule owns its eligibility, special targeting range and restoration continuation.</summary>
    public static void QueueGrantedAttack(GameContext game, int attacker, int? defender = null,
        int? forcedBlueDice = null, GamePhase returnPhase = GamePhase.AbilityChoice,
        bool flakOnly = false, int? forcedBlackDice = null, RangeBand? maximumRange = null)
    {
        if (attacker < 0 || attacker >= game.Topology.EntityCount) throw new ArgumentOutOfRangeException(nameof(attacker));
        game.Flow.Flags["combat.granted.active"] = attacker + 1;
        game.Flow.Flags["combat.granted.completed"] = 0;
        game.Flow.Flags["combat.granted.return"] = (int)returnPhase;
        if (forcedBlueDice.HasValue) game.Flow.Flags["combat.granted.blue"] = forcedBlueDice.Value;
        if (forcedBlackDice.HasValue) game.Flow.Flags["combat.granted.black"] = forcedBlackDice.Value;
        if (flakOnly) game.Flow.Flags["combat.granted.flak"] = 1;
        if (maximumRange.HasValue) game.Flow.Flags["combat.granted.maximum-range"] = (int)maximumRange.Value;
        if (game.IsShip(attacker)) { game.Flow.ActiveShip = attacker; game.Flow.ActiveSquadron = -1; }
        else { game.Flow.ActiveSquadron = attacker; game.Flow.SquadronCanAttack = true; }
        if (!defender.HasValue) { game.Flow.Phase = GamePhase.SelectAttack; return; }
        using var scope = new GameScope(game);
        var battery = forcedBlueDice.HasValue ? new DiceBattery(0, forcedBlueDice.Value, 0)
            : Battery(game, attacker, defender.Value, HullZone.Front, RangeBand.Close, HullZone.Front);
        var state = new AttackState { Attacker = attacker, Defender = defender.Value, FromZone = HullZone.Front,
            ToZone = HullZone.Front, Range = RangeBand.Close, Dice = Unrolled(battery), RedDice = battery.Red,
            BlueDice = battery.Blue, BlackDice = battery.Black, ResumePhase = returnPhase };
        if (game.Flow.Flags.ContainsKey("combat.granted.obstructed")) { state.Obstructed = true; state.MandatoryRemovals = 1; }
        state.EffectFlags["granted-attack"] = 1;
        game.Flow.Attacks.Add(state); new CombatModule().Enter(game, state, GamePhase.ModifyAttack, CombatWindow.BeforeRoll);
    }

    private GameDecision AttackSelection(GameContext game)
    {
        int attacker = game.Flow.ActiveSquadron >= 0 ? game.Flow.ActiveSquadron : game.Flow.ActiveShip;
        if (attacker < 0) throw new InvalidOperationException("No active combat unit.");
        var options = new List<GameActionOption>();
        bool ship = game.IsShip(attacker);
        if (ship) options.AddRange(CommandModule.ActivationCommandOptions(game, GamePhase.SelectAttack));
        bool extraFlak = game.Flow.Flags.ContainsKey("combat.granted.flak");
        bool mayAttack = game.IsOnBoard(attacker);
        HullZone? flak = ship ? FlakZone(game, attacker) : null;
        if (ship)
        {
            var activation = game.Flow.ShipActivations[attacker];
            int limit = UpgradeAttackDice.LimitsToOneAttackPerRound(game.ShipViews[attacker]) ? 1
                : game.State.Ships[attacker].HasAuxiliaryHullZones ? 3 : 2;
            mayAttack &= extraFlak || flak.HasValue || activation.AttacksResolved < limit;
            if (!extraFlak && UnitAbilityEngine.PatriotFistLimitsFurtherAttacks(game.ShipViews[attacker])) mayAttack = false;
        }
        else mayAttack &= game.Flow.SquadronCanAttack && !game.State.Squadrons[game.SquadronIndex(attacker)].TemporaryCannotAttack
            && !(MissingSquadronCombatEffects.Is(game, attacker, MissingSquadronCombatEffects.Vult) && !game.Flow.SquadronCommanded);
        if (mayAttack && ship && !extraFlak && !flak.HasValue && game.Flow.ShipActivations[attacker].AttacksResolved == 0)
            for (int defender = 0; defender < game.State.Ships.Length; defender++)
            {
                if (defender == attacker || Owner(game, defender) == Owner(game, attacker) || !game.IsOnBoard(defender)) continue;
                foreach (var to in TargetZones(game, defender))
                    if (IgnitionAttackGeometry.TryMeasureAttack(game, attacker, defender, to, out var measure))
                        options.Add(Option(new(GameActionKind.Attack, attacker, defender, measure.AssociatedHullZone, to,
                            AbilityId: "attack.ignition"), $"Ignition {measure.AssociatedHullZone} attacks entity {defender} {to}"));
            }
        // The upstream controller requires a legal prepared ignition attack to be the first attack.
        if (options.Any(option => option.Template.AbilityId == "attack.ignition")) return new(DecisionKind.Player, Owner(game, attacker), game.Flow.Phase, options);
        if (mayAttack)
            for (int defender = 0; defender < game.Topology.EntityCount; defender++)
            {
                if (defender == attacker || Owner(game, defender) == Owner(game, attacker) || !game.IsOnBoard(defender)) continue;
                if (extraFlak && game.IsShip(defender)) continue;
                if (flak.HasValue && (game.IsShip(defender) || game.Flow.Flags.ContainsKey(FlakBatchTargetKey(game, attacker, defender)))) continue;
                foreach (var from in ship ? game.State.Ships[attacker].HullZones : new[] { HullZone.Front })
                {
                    if (ship && game.Flow.Flags.ContainsKey(SourceCombatEffects.AckbarKey(game, attacker)) && from is not (HullZone.Left or HullZone.Right)) continue;
                    if (flak.HasValue && from != flak.Value) continue;
                    if (ship && !extraFlak && !flak.HasValue && (game.Flow.ShipActivations[attacker].UsedHullZones & (1 << (int)from)) != 0
                        && !CanReuseZone(game, attacker, from)) continue;
                    if (ship && game.HasUpgrade(attacker, "gunnery-team") && CanReuseZone(game, attacker, from) && (game.IsShip(defender)
                        ? game.Flow.ShipActivations[attacker].AttackedShips.Contains(defender)
                        : game.Flow.Flags.ContainsKey(FlakTargetKey(game, attacker, defender, from)))) continue;
                    if (ship && game.IsShip(defender) && !game.State.Ships[attacker].CanPerformSecondAttackAgainstShip
                        && game.Flow.ShipActivations[attacker].AttackedShips.Length > 0) continue;
                    foreach (var to in TargetZones(game, defender))
                        if ((!ship || OfficialAssaultObjectives.AllowsTarget(game, attacker, defender, from, to)) && TryAttackGeometry(game, attacker, defender, from, to, out var candidateRange, out _, out _)
                            && (!game.Flow.Flags.TryGetValue("combat.granted.maximum-range", out long maximum) || (int)candidateRange <= maximum))
                            options.Add(Option(new(GameActionKind.Attack, attacker, defender, from, to), $"{from} attacks entity {defender} {to}"));
                }
            }
        if (mayAttack && ship && !flak.HasValue)
            for (int index = 0; index < game.State.SpatialObjects.Count; index++)
                foreach (var from in game.State.Ships[attacker].HullZones)
                    if ((!game.Flow.Flags.ContainsKey(SourceCombatEffects.AckbarKey(game, attacker)) || from is HullZone.Left or HullZone.Right)
                        && (extraFlak || (game.Flow.ShipActivations[attacker].UsedHullZones & (1 << (int)from)) == 0 || CanReuseZone(game, attacker, from))
                        && TryMineAttackGeometry(game, attacker, index, from, out _, out _, out _))
                        options.Add(Option(new(GameActionKind.Attack, attacker, FromZone: from, Index: index, AbilityId: "attack.mine"), "Shoot mine " + index + " from " + from));
        if (extraFlak) options.Add(Option(new(GameActionKind.SkipAttack, attacker), "Finish additional anti-squadron attack"));
        else if (ship && game.Flow.Flags.ContainsKey($"combat.demolisher.after-maneuver.{game.State.Round}.{attacker}"))
            options.Add(Option(new(GameActionKind.SkipAttack, attacker), "Decline Demolisher attack"));
        else if (ship && game.Flow.ShipActivations[attacker].ManeuverComplete && !flak.HasValue)
            options.Add(Option(new(GameActionKind.EndActivation, attacker), "End ship activation"));
        else options.Add(Option(new(GameActionKind.SkipAttack, attacker), flak.HasValue ? "Finish anti-squadron attack" : "Finish attacks"));
        return new(DecisionKind.Player, Owner(game, attacker), game.Flow.Phase, options);
    }

    public static bool TryAttackGeometry(GameContext game, int attacker, int defender, HullZone from, HullZone to,
        out RangeBand range, out bool obstructed, out float distance)
    {
        using var scope = new GameScope(game);
        range = RangeBand.OutOfRange; obstructed = false; distance = float.PositiveInfinity;
        if (!game.IsOnBoard(attacker) || !game.IsOnBoard(defender)) return false;
        Segment line;
        if (game.IsShip(attacker))
        {
            var attackZone = Zone(game, attacker, from);
            if (game.IsShip(defender))
            {
                if (Station(game, defender) && (to != HullZone.Front || Geometry2D.Overlaps(game.ShipBase(attacker), StationShape(game, defender)))) return false;
                var target = Station(game, defender) ? StationZone(game, defender, attackZone.TargetingPoint) : Zone(game, defender, to);
                var measurement = HullGeometry.MeasureAttack(attackZone, target);
                if (!measurement.HasEligibleDefenderEdge || !Station(game, defender) && !HullGeometry.IsTargetZoneExposed(attackZone.TargetingPoint, target, game.ShipBase(defender))) return false;
                distance = measurement.Distance; line = new(attackZone.TargetingPoint, target.TargetingPoint);
            }
            else
            {
                var target = game.SquadronBase(defender);
                if (!attackZone.ContainsInArc(target.Center) && !ArcIntersectsCircle(attackZone, target)) return false;
                distance = HullGeometry.MeasureToSquadron(attackZone, target).Distance;
                line = new(attackZone.TargetingPoint, target.Center);
            }
            range = HullGeometry.ClassifyRange(distance);
            if (!game.State.Ships[attacker].CanAttackAtRange(range)) return false;
        }
        else
        {
            var attackerBase = game.SquadronBase(attacker);
            if (game.IsShip(defender))
            {
                if (Station(game, defender) && (to != HullZone.Front || Geometry2D.Overlaps(attackerBase, StationShape(game, defender)))) return false;
                var target = Station(game, defender) ? StationZone(game, defender, attackerBase.Center) : Zone(game, defender, to);
                distance = target.Edges.ToArray().Min(edge => Math.Max(0, Vector2.Distance(attackerBase.Center, Geometry2D.ClosestPoint(edge, attackerBase.Center)) - attackerBase.Radius));
                if (!Station(game, defender) && !HullGeometry.IsTargetZoneExposed(attackerBase.Center, target, game.ShipBase(defender))) return false;
                line = new(attackerBase.Center, target.TargetingPoint);
                if (!SquadronAbilityEngine.CanAttackShip(game.SquadronViews[game.SquadronIndex(attacker)], game.SceneSquadronViews)) return false;
            }
            else
            {
                distance = Geometry2D.Distance(attackerBase, game.SquadronBase(defender));
                line = new(attackerBase.Center, game.SquadronBase(defender).Center);
                if (!SquadronAbilityEngine.CanAttackSquadron(game.SquadronViews[game.SquadronIndex(attacker)],
                    game.SquadronViews[game.SquadronIndex(defender)], game.SceneSquadronViews,
                    distance / TabletopMeasurements.MillimetresPerWorldUnit, out _)) return false;
            }
            var view = game.SquadronViews[game.SquadronIndex(attacker)];
            float allowedRange = game.IsShip(defender) ? SquadronAbilityEngine.GetShipAttackDistance(view, game.SceneSquadronViews)
                : SquadronAbilityEngine.GetSquadronAttackDistance(view);
            if (distance > TabletopMeasurements.FromWorldUnits(allowedRange)) return false;
            range = RangeBand.Close;
        }
        var blockers = new List<LineOfSightObstacle>();
        for (int ship = 0; ship < game.State.Ships.Length; ship++)
            if (ship != attacker && ship != defender && !Station(game, ship) && game.IsOnBoard(ship)) blockers.Add(new(game.ShipBase(ship)));
        for (int index = 0; index < game.State.SpatialObjects.Count; index++)
        {
            var item = game.State.SpatialObjects[index];
            if (OfficialObjectives.Is(game, "contested-outpost") && OfficialObjectives.ObstacleId(game, index) == "station") continue;
            if (OfficialObjectives.State(game)?.PieceEntities.TryGetValue(index, out int stationEntity) == true && (stationEntity == attacker || stationEntity == defender)) continue;
            if (item.Active && item.Kind == (int)Armada.Session.MapPlacementKind.Obstacle) blockers.Add(new(ObstacleGeometry.Shape(game, index)));
        }
        obstructed = LineOfSightGeometry.Evaluate(line, blockers.ToArray()) == LineOfSightResult.Obstructed;
        if (game.IsShip(defender))
        {
            obstructed |= CommanderRules.DefenderIsTreatedAsObstructed(game.ShipViews[defender]).Applied
                || OfficerCombatRules.ObstructionNotice(game.ShipViews[defender]).Applied;
            if (game.Flow.Flags.GetValueOrDefault($"round.early-warning.{defender}") == (int)to + 1) obstructed = true;
        }
        else obstructed |= SquadronAbilityEngine.TreatsDefenderAsObstructed(game.SquadronViews[game.SquadronIndex(defender)])
            || TitleRules.StrongholdSource(game.SquadronViews[game.SquadronIndex(defender)]) is not null;
        if (!game.IsShip(attacker) && !game.IsShip(defender))
        {
            var source = game.SquadronViews[game.SquadronIndex(attacker)]; var target = game.SquadronViews[game.SquadronIndex(defender)];
            obstructed |= CloneWarsSquadronRules.LuminaraObstructs(source, target, game.SceneSquadronViews)
                || SquadronUpgradeRules.JammingFieldSource(source, target) is not null;
        }
        if (game.IsShip(attacker) && UnitAbilityEngine.AttacksCannotBeObstructed(game.ShipViews[attacker])) obstructed = false;
        if (!game.IsShip(attacker) && CloneWarsSquadronRules.AttacksCannotBeObstructed(game.SquadronViews[game.SquadronIndex(attacker)])) obstructed = false;
        if (game.IsShip(attacker) && obstructed && !game.State.Ships[attacker].CanChooseObstructedTarget) return false;
        var battery = Battery(game, attacker, defender, from, range, to);
        int count = game.Flow.Flags.TryGetValue("combat.granted.black", out long forcedBlack) ? (int)forcedBlack : battery.Total;
        return range != RangeBand.OutOfRange && count > 0;
    }

    private void BeginAttack(GameContext game, GameAction action)
    {
        if (!game.IsShip(action.Source) && game.Flow.SquadronCommanded) CommandModule.CommitSquadronSelection(game, action.Source);
        if (action.AbilityId == "attack.mine") { BeginMineAttack(game, action); return; }
        ValidateUpgrades(game, action.Source, action.Target);
        RangeBand range; bool obstructed; float distance; DiceBattery battery; bool ignition = action.AbilityId == "attack.ignition";
        IgnitionAttackMeasurement ignitionMeasure = default;
        if (ignition)
        {
            if (!IgnitionAttackGeometry.TryMeasureAttack(game, action.Source, action.Target, action.ToZone, out ignitionMeasure)
                || action.FromZone != ignitionMeasure.AssociatedHullZone) throw new ArgumentException("Ignition geometry is no longer legal.", nameof(action));
            range = ignitionMeasure.AttackRange; obstructed = ignitionMeasure.Obstructed; distance = ignitionMeasure.AttackDistance;
            var increase = UpgradeAttackDice.BatteryArmamentIncrease(game.ShipViews[action.Source], action.FromZone);
            var printed = ignitionMeasure.Battery; var diceRange = ignitionMeasure.DiceRange;
            bool disposable = game.Flow.Flags.GetValueOrDefault($"activation.disposable-capacitors.{game.State.Round}.{action.Source}") != 0;
            battery = new(diceRange == RangeBand.OutOfRange ? 0 : printed.RedDice + increase.RedDice,
                diceRange == RangeBand.OutOfRange || diceRange == RangeBand.Long && !disposable ? 0 : printed.BlueDice + increase.BlueDice,
                diceRange == RangeBand.Close || diceRange == RangeBand.Medium && TitleRules.AllowsBlackDiceAtMediumAgainstRear(game.ShipViews[action.Source], action.ToZone)
                    ? printed.BlackDice + increase.BlackDice : 0);
        }
        else
        {
            if (!TryAttackGeometry(game, action.Source, action.Target, action.FromZone, action.ToZone, out range, out obstructed, out distance))
                throw new ArgumentException("Attack geometry is no longer legal.", nameof(action));
            battery = Battery(game, action.Source, action.Target, action.FromZone, range, action.ToZone);
        }
        var dice = Unrolled(battery);
        var a = new AttackState { Attacker = action.Source, Defender = action.Target, FromZone = action.FromZone,
            ToZone = action.ToZone, Range = range, Obstructed = obstructed,
            IgnitionAttack = ignition, ExtremeRange = ignition && ignitionMeasure.IsExtremeRange,
            DiceRange = ignition ? ignitionMeasure.DiceRange : range,
            RedDice = battery.Red, BlueDice = battery.Blue, BlackDice = battery.Black, Dice = dice,
            ResumePhase = game.IsShip(action.Source) ? GamePhase.SelectAttack : GamePhase.SquadronActivation,
            MandatoryRemovals = obstructed && !(game.IsShip(action.Source)
                && CommanderRules.ObstructionBecomesAdvantage(game.ShipViews[action.Source]).Applied) ? 1 : 0 };
        if (game.Flow.Flags.ContainsKey("combat.granted.active"))
        {
            a.EffectFlags["granted-attack"] = 1; a.ResumePhase = (GamePhase)game.Flow.Flags["combat.granted.return"];
            a.EffectFlags["granted-natural-attack"] = 1;
            if (game.Flow.Flags.TryGetValue("combat.granted.blue", out long blue)) a.Dice = Unrolled(new(0, (int)blue, 0));
            if (game.Flow.Flags.TryGetValue("combat.granted.black", out long black)) a.Dice = Unrolled(new(0, 0, (int)black));
            if (game.Flow.Flags.ContainsKey("combat.granted.flak") && !game.Flow.Flags.ContainsKey("combat.granted.single-target"))
            { a.EffectFlags["granted-flak"] = 1; a.ResumePhase = GamePhase.SelectAttack; }
            if (game.Flow.Flags.TryGetValue("combat.granted.upgrade-index", out long upgradeIndex) && !game.Flow.Flags.ContainsKey("combat.granted.paid"))
            {
                var card = game.State.Upgrades[(int)upgradeIndex];
                if (!(game.Flow.Flags.ContainsKey("combat.granted.exhaust") ? card.Exhaust() : card.TryUseThisActivation()))
                    throw new InvalidOperationException("The granting upgrade is no longer available.");
                game.Flow.Flags["combat.granted.paid"] = 1;
            }
        }
        if (game.Flow.Flags.ContainsKey($"combat.demolisher.after-maneuver.{game.State.Round}.{action.Source}"))
            a.ResumePhase = game.IsShip(action.Target) ? GamePhase.AbilityChoice : GamePhase.SelectAttack;
        if (!game.IsShip(action.Source))
        {
            a.IgnoreShields = SquadronAbilityEngine.IgnoresShipShields(game.SquadronViews[game.SquadronIndex(action.Source)]);
            a.SnipeAttack = !game.IsShip(action.Target) && SquadronAbilityEngine.IsSnipeAttack(game.SquadronViews[game.SquadronIndex(action.Source)],
                distance / TabletopMeasurements.MillimetresPerWorldUnit);
        }
        if (game.IsShip(action.Source))
        {
            var ship = game.State.Ships[action.Source];
            if (game.IsShip(action.Target) ? ship.MustRemoveDieWhenAttackingShip : ship.MustRemoveDieWhenAttackingSquadron) a.MandatoryRemovals++;
            if (!game.IsShip(action.Target))
            {
                if (!FlakZone(game, action.Source).HasValue)
                {
                    string sequence = FlakKey(game, action.Source) + ".sequence";
                    game.Flow.Flags[sequence] = game.Flow.Flags.GetValueOrDefault(sequence) + 1;
                    game.Flow.Flags[FlakKey(game, action.Source)] = (int)action.FromZone + 1;
                    if (!game.Flow.Flags.ContainsKey("combat.granted.flak"))
                    { var activation = game.Flow.ShipActivations[action.Source]; activation.AttacksResolved++; activation.UsedHullZones |= 1 << (int)action.FromZone; }
                }
                game.Flow.Flags[FlakTargetKey(game, action.Source, action.Target)] = 1;
                game.Flow.Flags[FlakBatchTargetKey(game, action.Source, action.Target)] = 1;
            }
        }
        foreach (var card in game.UpgradesFor(action.Source)) card.ResetAttackUsage();
        foreach (var card in game.UpgradesFor(action.Target)) card.ResetAttackUsage();
        if (!game.IsShip(action.Source)) ApplyAutomaticAttackBonuses(game, a);
        game.Flow.Attacks.Add(a); Enter(game, a, GamePhase.ModifyAttack, CombatWindow.BeforeRoll);
    }

    internal static DiceBattery Battery(GameContext game, int attacker, int defender, HullZone zone, RangeBand range, HullZone targetZone)
    {
        BatteryArmament printed;
        if (game.IsShip(attacker))
        {
            var def = game.Topology.Ships[attacker].CatalogEntry.Definition;
            printed = game.IsShip(defender) ? UpgradeAttackDice.EffectiveBatteryArmament(game.ShipViews[attacker], zone)
                : TurbolaserRules.EffectiveFlakArmament(game.ShipViews[attacker], game.SquadronViews[game.SquadronIndex(defender)],
                    OffensiveRetrofitRules.EffectiveAntiSquadronArmament(game.ShipViews[attacker], def.AntiSquadronArmament));
        }
        else
        {
            var def = game.Topology.Squadrons[game.SquadronIndex(attacker)].CatalogEntry.Definition;
            printed = game.IsShip(defender) ? def.AntiShipArmament : def.AntiSquadronArmament;
            if (!game.IsShip(defender))
            {
                var source = game.SquadronViews[game.SquadronIndex(attacker)];
                if (SquadronAbilityEngine.IsSnipeAttack(source, SquadronAbilityEngine.FootprintDistance(source, game.SquadronViews[game.SquadronIndex(defender)])))
                    printed = new(0, SquadronAbilityEngine.GetSnipeValue(source), 0);
            }
        }
        int red = printed.RedDice, blue = printed.BlueDice, black = printed.BlackDice;
        if (game.IsShip(attacker))
        {
            if (range == RangeBand.Long && game.Flow.Flags.GetValueOrDefault($"activation.disposable-capacitors.{game.State.Round}.{attacker}") == 0) blue = 0;
            if (range != RangeBand.Close && !(game.IsShip(defender) && range == RangeBand.Medium
                && TitleRules.AllowsBlackDiceAtMediumAgainstRear(game.ShipViews[attacker], targetZone))) black = 0;
        }
        return new(red, blue, black);
    }

    private static void ApplyAutomaticAttackBonuses(GameContext game, AttackState a)
    {
        if (a.EffectFlags.ContainsKey("automatic-bonuses.applied")) return;
        a.EffectFlags["automatic-bonuses.applied"] = 1;
        if (a.Counter || a.IsSalvoAttack || a.Attacker < 0) return;
        var added = new List<AttackDie>();
        void Add(AttackDieColor color, int count) { for (int i = 0; i < count; i++) added.Add(new(color, DieFace.Blank)); }
        if (game.IsShip(a.Attacker))
        {
            if (game.IsShip(a.Defender))
            {
                var activation = game.Flow.ShipActivations[a.Attacker];
                var bonus = UpgradeAttackDice.Evaluate(new(game.ShipViews[a.Attacker], game.ShipViews[a.Defender], a.FromZone,
                    a.ToZone, a.Range, activation.AttackedShips.Contains(a.Defender), game.State.Activated[a.Defender], game.State.Round, activation.AttacksResolved));
                var increase = UpgradeAttackDice.BatteryArmamentIncrease(game.ShipViews[a.Attacker], a.FromZone);
                Add(AttackDieColor.Red, Math.Max(0, bonus.Red - increase.RedDice));
                Add(AttackDieColor.Blue, Math.Max(0, bonus.Blue - increase.BlueDice));
                Add(AttackDieColor.Black, Math.Max(0, bonus.Black - increase.BlackDice));
                if (game.Flow.Flags.ContainsKey(SourceCombatEffects.AckbarKey(game, a.Attacker))
                    && !a.EffectFlags.ContainsKey("used.commander.admiral-ackbar.commit-broadsides")) Add(AttackDieColor.Red, 2);
                if (a.Obstructed && CommanderRules.ObstructionBecomesAdvantage(game.ShipViews[a.Attacker]).Applied) Add(AttackDieColor.Red, 1);
            }
        }
        else if (!game.IsShip(a.Defender))
        {
            var source = game.SquadronViews[game.SquadronIndex(a.Attacker)]; var target = game.SquadronViews[game.SquadronIndex(a.Defender)];
            Add(AttackDieColor.Blue, SquadronAbilityEngine.GetBonusBlueDice(source, target, game.SceneSquadronViews));
            if (CommandUpgradeRules.GrantsFlightControllersBlueDie(source)) Add(AttackDieColor.Blue, 1);
        }
        a.Dice = [..a.Dice, ..added];
    }

    private void ApplyRoll(GameContext game, AttackState a, GameAction action)
    {
        if (action.Kind != GameActionKind.ResolveDice || action.Dice is null || action.Dice.Length != a.Dice.Length)
            throw new ArgumentException("A chance action must provide the complete pending dice pool.", nameof(action));
        var indices = a.PendingRollIndices.ToHashSet();
        for (int i = 0; i < a.Dice.Length; i++)
        {
            var die = action.Dice[i];
            if (die.Color != a.Dice[i].Color || !Faces(die.Color).Contains(die.Face)
                || (!indices.Contains(i) && die.Face != a.Dice[i].Face))
                throw new ArgumentException("Illegal dice chance outcome.", nameof(action));
        }
        a.Dice = action.Dice.ToArray(); a.PendingRollIndices = [];
        if (a.EnvironmentalDamage && (a.DamageRule.StartsWith("mine.explosion.", StringComparison.Ordinal) || a.DamageRule == "cluster-bombs.explosion"))
        {
            a.Damage = a.Dice.Count(die => die.Face is DieFace.Hit or DieFace.Critical or DieFace.HitCritical);
            if (a.DamageRule == "mine.explosion.objective" && a.Dice.Any(die => die.Face is DieFace.Critical or DieFace.HitCritical)) a.FirstHullCardFaceup = true;
            a.Rolled = true; game.Flow.Phase = GamePhase.DealDamage; return;
        }
        bool firstRoll = !a.Rolled; a.Rolled = true;
        if (firstRoll && a.Attacker >= 0 && !game.IsShip(a.Attacker) && !a.Counter)
        {
            if (game.IsShip(a.Defender) && CommanderRules.GrantsPloKoonAccuracy(game.SquadronViews[game.SquadronIndex(a.Attacker)], game.ShipViews[a.Defender]))
                a.Dice = [..a.Dice, new AttackDie(AttackDieColor.Blue, DieFace.Accuracy)];
        }
        if (firstRoll && a.IgnitionAttack) IgnitionAttackGeometry.ConsumeTarget(game, a.Attacker);
        if (firstRoll && OfficialNavigationObjectives.SolarCoronaOptions(game, a).Any())
        { a.EffectFlags["objective.solar-pending"] = 1; a.Window = CombatWindow.AttackEffects; game.Flow.Phase = GamePhase.ModifyAttack; }
        else if (firstRoll) Enter(game, a, GamePhase.ModifyAttack, CombatWindow.AttackEffects);
        else game.Flow.Phase = a.AfterRollPhase;
        effects.AfterDiceChance(game, a);
    }

    private static bool Locked(AttackState a, DefenseTokenInstance token)
        => a.LockedTokens.Any(locked => locked.Token == token.Token && locked.Slot == token.Slot);

    private static IEnumerable<DefenseTokenInstance> Tokens(GameContext game, int defender)
    {
        foreach (var type in Enum.GetValues<DefenseTokenType>())
            foreach (var token in game.IsShip(defender) ? game.State.Ships[defender].GetDefenseTokenInstances(type)
                : game.State.Squadrons[game.SquadronIndex(defender)].GetDefenseTokenInstances(type)) yield return token;
    }

    internal static bool CanSpend(GameContext game, AttackState a, DefenseTokenInstance token, bool ignoreAccuracy = false)
    {
        if ((!ignoreAccuracy && Locked(a, token)) || a.SpentTokenTypes.Contains(token.Token)) return false;
        if (token.Token == DefenseTokenType.Redirect && a.EffectFlags.ContainsKey("barriss.redirect-spent")) return false;
        if (token.Exhausted && a.EffectFlags.GetValueOrDefault("avenger") != 0 && a.EffectFlags.GetValueOrDefault("exhausted-token-spends") > 0) return false;
        if (game.IsShip(a.Attacker) && a.SpentTokenTypes.Length > 0
            && TitleRules.LimitsDefenseTokens(game.ShipViews[a.Attacker],
                a.EffectFlags.ContainsKey(ConcentrateFireMode))) return false;
        if (game.IsShip(a.Defender) && a.SpentTokenTypes.Length > 0
            && game.ShipViews[a.Defender].Loadout.Find("jedi-hostage") is { Active: true, Exhausted: true }) return false;
        if (token.Token == DefenseTokenType.Salvo && (a.IsSalvoAttack || a.Counter || !IgnitionDefenseRules.AllowsSalvo(a.IgnitionAttack))) return false;
        if (!game.IsShip(a.Defender)) return token.Token is DefenseTokenType.Brace or DefenseTokenType.Scatter or DefenseTokenType.Evade;
        var ship = game.State.Ships[a.Defender];
        if (ship.CurrentSpeed == 0 && !(a.ToZone == HullZone.Front && game.HasUpgrade(a.Defender, "concord-title"))) return false;
        if (token.Exhausted && !ship.CanSpendExhaustedDefenseTokens) return false;
        if (token.Token == DefenseTokenType.Redirect && ship.HasFaceupCritical(CriticalDamageCard.CapacitorFailure) && ship.GetShields(a.ToZone) == 0) return false;
        return true;
    }

    internal static void SpendToken(GameContext game, AttackState a, GameAction action)
    {
        var token = Tokens(game, a.Defender).Single(x => x.Token == action.Token && x.Slot == action.Index);
        bool discarded;
        bool spent = game.IsShip(a.Defender) ? game.State.Ships[a.Defender].TryResolveDefenseToken(token, out discarded)
            : game.State.Squadrons[game.SquadronIndex(a.Defender)].TryResolveDefenseToken(token, out discarded);
        if (!spent) throw new ArgumentException("Defense token is unavailable.", nameof(action));
        a.EffectFlags["defender.spent-token"] = 1;
        if (discarded) a.EffectFlags["exhausted-token-spends"] = a.EffectFlags.GetValueOrDefault("exhausted-token-spends") + 1;
        a.SpentTokenTypes = [..a.SpentTokenTypes, token.Token];
        if (a.IntelOfficerTarget is { } intelTarget && IntelOfficerTargetRules.MatchesSpend(intelTarget,
            game.IsShip(a.Defender) ? RuntimeUnitKind.Ship : RuntimeUnitKind.Squadron,
            game.IsShip(a.Defender) ? game.ShipViews[a.Defender].RuntimeId : game.SquadronViews[game.SquadronIndex(a.Defender)].RuntimeId, token))
        {
            if (game.IsShip(a.Defender)) IntelOfficerTargetRules.TryDiscardAfterSpend(game.State.Ships[a.Defender], token);
            else IntelOfficerTargetRules.TryDiscardAfterSpend(game.State.Squadrons[game.SquadronIndex(a.Defender)], token);
            discarded = true;
        }
        if (!discarded && RemainingUpgradeRules.PalpatinePunishesToken(Owner(game, a.Defender), token.Token))
        {
            var exhausted = new DefenseTokenInstance(token.Token, token.Slot, true);
            if (game.IsShip(a.Defender)) game.State.Ships[a.Defender].DiscardDefenseToken(exhausted);
            else game.State.Squadrons[game.SquadronIndex(a.Defender)].DiscardDefenseToken(exhausted);
            discarded = true;
        }
        if (discarded) a.EffectFlags["defender.discarded-token"] = 1;
        switch (token.Token)
        {
            case DefenseTokenType.Brace: a.Brace = true; break;
            case DefenseTokenType.Scatter:
                a.Scatter = true;
                MissingSquadronCombatEffects.RemoveDice(a, MissingSquadronCombatEffects.ModifiableDice(a)); break;
            case DefenseTokenType.Contain: a.Contain = true; break;
            case DefenseTokenType.Redirect: a.Redirect = true; break;
            case DefenseTokenType.Salvo: a.Salvo = true; break;
            case DefenseTokenType.Evade:
                a.Evade = true; a.SpentEvade = token; a.EvadeWasDiscarded = discarded;
                bool mothma = FriendlyUpgrade(game, a.Defender, "mon-mothma-commander");
                a.EvadeBaseCount = game.HasUpgrade(a.Defender, "foresight-title") ? 2 : 1;
                if (a.ExtremeRange) a.EvadeBaseCount = IgnitionDefenseRules.ExtremeEvadeCancellationCount(game.HasUpgrade(a.Defender, "foresight-title"));
                if (mothma && a.Range == RangeBand.Close) a.EvadeBaseCount++;
                a.EvadeCancels = a.ExtremeRange || a.Range == RangeBand.Long || (mothma && a.Range == RangeBand.Medium)
                    || (a.Range == RangeBand.Close && game.HasUpgrade(a.Defender, "iden-versio-officer"));
                if (!game.IsShip(a.Defender)) { a.EvadeBaseCount = 1; a.EvadeCancels = false; }
                if (a.Dice.Length > 0) game.Flow.Phase = GamePhase.EvadeDice;
                break;
        }
    }

    private static void ApplyEvade(GameContext game, AttackState a, GameAction action)
    {
        int[] indices = action.DiceIndices!;
        if (indices.Length > a.EvadeBaseCount && !a.EvadeWasDiscarded)
        {
            var spent = a.SpentEvade!.Value;
            if (!game.State.Ships[a.Defender].DiscardDefenseToken(new DefenseTokenInstance(spent.Token, spent.Slot, exhausted: true)))
                throw new InvalidOperationException("Cannot discard the spent Evade token.");
            a.EvadeWasDiscarded = true;
        }
        if (a.EvadeCancels)
        {
            MissingSquadronCombatEffects.RemoveDice(a, indices);
            game.Flow.Phase = GamePhase.SpendDefense;
        }
        else
        {
            a.PendingRollIndices = indices.ToArray(); a.AfterRollPhase = GamePhase.SpendDefense;
            game.Flow.Phase = GamePhase.RollAttack;
        }
    }

    private void FinishDefense(GameContext game, AttackState a)
    {
        a.EffectFlags["defense.finished"] = 1;
        if (effects.TryFinishWindow(game, a, CombatWindow.DefenseEffects)) return;
        int damage = Damage(game, a);
        damage = Math.Max(0, damage - (int)a.EffectFlags.GetValueOrDefault("prebrace.damage-reduction"));
        if (a.Brace && !(game.HasUpgrade(a.Attacker, "heavy-turbolaser-turrets") && a.SpentTokenTypes.Length > 1)) damage = (damage + 1) / 2;
        if (game.HasUpgrade(a.Defender, "bright-hope-title") && a.ToZone != HullZone.Rear && damage > 0) damage--;
        if (!game.IsShip(a.Defender)) TitleRules.ReduceSquadronAttackDamage(game.SquadronViews[game.SquadronIndex(a.Defender)], ref damage);
        a.Damage = damage;
        if (a.Redirect && game.IsShip(a.Defender) && damage > 0) game.Flow.Phase = GamePhase.RedirectDamage;
        else Enter(game, a, GamePhase.ResolveCritical, CombatWindow.CriticalEffects);
    }

    private static int Damage(GameContext game, AttackState a)
    {
        int hits = 0, criticals = 0, accuracies = 0, blanks = 0;
        foreach (var die in a.Dice)
            switch (die.Face)
            {
                case DieFace.Hit: hits++; break;
                case DieFace.DoubleHit: hits += 2; break;
                case DieFace.Critical: criticals++; break;
                case DieFace.HitCritical: hits++; criticals++; break;
                case DieFace.Accuracy: accuracies++; break;
                case DieFace.Blank: blanks++; break;
            }
        if (game.IsShip(a.Defender))
        {
            if (a.EffectFlags.ContainsKey("neutral.ship")) hits += criticals;
            else if (game.IsShip(a.Attacker)) hits += criticals * UnitAbilityEngine.GetCriticalDamagePerIcon(game.ShipViews[a.Attacker], a.FromZone)
                + IonCannonRules.UnspentAccuracyDamage(game.ShipViews[a.Attacker], accuracies);
            else
            {
                var squad = game.SquadronViews[game.SquadronIndex(a.Attacker)];
                if (SquadronAbilityEngine.HasBomber(squad)) hits += criticals;
                hits += CloneWarsSquadronRules.GrievousCriticalDamageBonus(squad, criticals, CloneWarsSquadronRules.HasReadyDefenseToken(game.ShipViews[a.Defender]));
            }
        }
        else if (a.Attacker >= 0 && !game.IsShip(a.Attacker))
            hits = SquadronAbilityEngine.DamageAgainstSquadron(game.SquadronViews[game.SquadronIndex(a.Attacker)],
                game.SquadronViews[game.SquadronIndex(a.Defender)], new(a.Range, a.Dice.Length, hits, criticals, accuracies, blanks, true, dice: a.Dice), a.Counter);
        if (!game.IsShip(a.Attacker) && MissingSquadronCombatEffects.Is(game, a.Attacker, MissingSquadronCombatEffects.Vader)) hits += criticals;
        return hits;
    }

    private static bool CanStandardCritical(GameContext game, AttackState a)
        => game.IsShip(a.Defender) && !Station(game, a.Defender) && !a.FirstHullCardFaceup && !a.Contain && a.Dice.Any(die => die.Face is DieFace.Critical or DieFace.HitCritical)
            && (!a.EffectFlags.ContainsKey("neutral.planetary") || game.State.Ships[a.Defender].GetShields(a.ToZone) == 0)
            && !game.HasUpgrade(a.Defender, "commander-gherant-officer")
            && (a.EffectFlags.ContainsKey("neutral.ship") || (game.IsShip(a.Attacker) ? game.State.Ships[a.Attacker].CanResolveCriticalEffects : SquadronAbilityEngine.HasBomber(game.SquadronViews[game.SquadronIndex(a.Attacker)])));

    private void DealDamage(GameContext game, AttackState a)
    {
        if (effects.TryFinishWindow(game, a, CombatWindow.BeforeDamage)) return;
        if (!game.IsShip(a.Defender))
        {
            int applied = game.State.Squadrons[game.SquadronIndex(a.Defender)].ApplyDamage(a.Damage);
            a.AppliedDamage += applied;
            if (applied > 0) EngineEffects.NotifyDamageApplied(game, a.Attacker, a.Defender, GamePhase.DealDamage);
            if (applied > 0 && a.EffectFlags.ContainsKey("neutral.exogorth"))
            {
                var squadron = game.State.Squadrons[game.SquadronIndex(a.Defender)];
                if (!squadron.Activated) squadron.MarkActivated(false);
                game.State.Activated[a.Defender] = true;
            }
            if (!a.EnvironmentalDamage && a.Attacker >= 0 && !game.IsShip(a.Attacker))
                SquadronAbilityEngine.ResolveDutchVander(game.SquadronViews[game.SquadronIndex(a.Attacker)], game.SquadronViews[game.SquadronIndex(a.Defender)], applied);
            CompleteAttack(game, a); return;
        }
        var ship = game.State.Ships[a.Defender];
        foreach (var entry in a.RedirectAllocations) a.AppliedDamage += ship.LoseShields(entry.Key, entry.Value);
        int shields = a.IgnoreShields ? 0 : ship.LoseShields(a.ToZone, a.Damage);
        a.AppliedDamage += shields;
        a.PendingHullCards = Math.Min(ship.CurrentHull, Math.Max(0, a.Damage - shields));
        a.DamageStarted = true;
        if (a.PendingHullCards > 0) game.Flow.Phase = GamePhase.ChooseDamageCard;
        else CompleteAttack(game, a);
    }

    private static bool Dodonna(GameContext game, AttackState a)
        => a.Attacker >= 0 && a.HullCardsDealt < Math.Max(a.FirstHullCardFaceup ? 1 : 0, a.EffectFlags.GetValueOrDefault("faceup.limit"))
            && FriendlyUpgrade(game, a.Attacker, "general-dodonna-commander");

    private void DrawCard(GameContext game, AttackState a, int choice)
    {
        var deck = Deck(game, a.Defender);
        var card = Dodonna(game, a) ? deck.ChooseAndDrawFromTop(choice, 4) : deck.Draw();
        bool faceup = a.HullCardsDealt < Math.Max(a.FirstHullCardFaceup ? 1 : 0, a.EffectFlags.GetValueOrDefault("faceup.limit"));
        a.HullCardsDealt++; a.PendingHullCards--;
        if (faceup)
        {
            a.IncomingDamageCard = card;
            if (IncomingCardOptions(game, a).Any(option => option.Template.AbilityId != "faceup-prevention.skip")) return;
            a.IncomingDamageCard = null;
        }
        CommitDamageCard(game, a, card, faceup);
    }

    private void CommitDamageCard(GameContext game, AttackState a, CriticalDamageCard card, bool faceup)
    {
        if (Station(game, a.Defender)) faceup = false;
        var ship = game.State.Ships[a.Defender];
        if (faceup) ship.DealFaceupDamageCard(card); else ship.DealFacedownDamageCard(card);
        if (faceup) OfficialAssaultObjectives.ObserveFaceupCard(game, a.Defender);
        a.AppliedDamage++;
        EngineEffects.NotifyDamageApplied(game, a.Attacker, a.Defender, GamePhase.ChooseDamageCard);
        if (faceup && card == CriticalDamageCard.LifeSupportFailure)
            foreach (var command in Enum.GetValues<ShipCommand>()) while (game.State.Commands[a.Defender].TrySpendToken(command)) { }
        if (faceup && CriticalEffectRegistry.Get(card).RequiresImmediatePrompt && !ship.Destroyed)
        {
            a.PendingDamageCard = card; a.CriticalChosenZones = []; game.Flow.Phase = GamePhase.ResolveCritical;
        }
        else ContinueHullDamage(game, a);
    }

    private GameDecision DamageCardDecision(GameContext game, AttackState a)
    {
        var ship = game.State.Ships[a.Defender]; var card = a.PendingDamageCard!.Value;
        var options = new List<GameActionOption>();
        int owner = Owner(game, a.Defender);
        switch (card)
        {
            case CriticalDamageCard.InjuredCrew:
                foreach (var token in Tokens(game, a.Defender))
                    options.Add(Option(new(GameActionKind.ChooseToken, Target: a.Defender, Token: token.Token, Index: token.Slot), "Discard " + token.Token + " " + token.Slot));
                break;
            case CriticalDamageCard.ProjectorMisaligned:
                int maximum = ship.HullZones.Max(ship.GetShields);
                foreach (var zone in ship.HullZones.Where(zone => ship.GetShields(zone) == maximum))
                    options.Add(Option(new(GameActionKind.ChooseZone, Target: a.Defender, ToZone: zone), "Lose all shields in " + zone));
                break;
            case CriticalDamageCard.ShieldFailure:
                owner = 3 - owner;
                if (a.CriticalChosenZones.Length < 2)
                    foreach (var zone in ship.HullZones.Where(zone => !a.CriticalChosenZones.Contains((int)zone)))
                        options.Add(Option(new(GameActionKind.ChooseZone, Target: a.Defender, ToZone: zone), "Lose one shield in " + zone));
                options.Add(Option(new(GameActionKind.Pass, Target: a.Defender), "Finish Shield Failure"));
                break;
            case CriticalDamageCard.CommNoise:
                owner = 3 - owner;
                if (ship.CurrentSpeed > 0) options.Add(Option(new(GameActionKind.ChangeSpeed, Target: a.Defender, Speed: ship.CurrentSpeed - 1), "Reduce speed by one"));
                if (game.State.Commands[a.Defender].Count > 0)
                    foreach (var command in Enum.GetValues<ShipCommand>())
                        options.Add(Option(new(GameActionKind.ChooseCommand, Target: a.Defender, Command: command), "Change top dial to " + command));
                options.Add(Option(new(GameActionKind.Pass, Target: a.Defender), "Leave speed and command unchanged"));
                break;
            case CriticalDamageCard.StructuralDamage:
                options.Add(Option(new(GameActionKind.ResolveDamage, Target: a.Defender, AbilityId: "critical.structural-damage"), "Suffer one facedown damage card"));
                break;
            default: throw new InvalidOperationException("No immediate critical decision handler: " + card);
        }
        if (options.Count == 0) options.Add(Option(new(GameActionKind.Pass, Target: a.Defender), "Finish critical effect"));
        return new(card == CriticalDamageCard.StructuralDamage ? DecisionKind.Forced : DecisionKind.Player,
            card == CriticalDamageCard.StructuralDamage ? 0 : owner, game.Flow.Phase, options);
    }

    private void ApplyDamageCardChoice(GameContext game, AttackState a, GameAction action)
    {
        var ship = game.State.Ships[a.Defender]; var card = a.PendingDamageCard!.Value;
        switch (card)
        {
            case CriticalDamageCard.InjuredCrew:
                if (action.Kind == GameActionKind.ChooseToken)
                    ship.DiscardDefenseToken(Tokens(game, a.Defender).Single(token => token.Token == action.Token && token.Slot == action.Index));
                break;
            case CriticalDamageCard.ProjectorMisaligned: ship.LoseAllShields(action.ToZone); break;
            case CriticalDamageCard.ShieldFailure:
                if (action.Kind == GameActionKind.ChooseZone)
                {
                    ship.LoseShields(action.ToZone, 1); a.CriticalChosenZones = [..a.CriticalChosenZones, (int)action.ToZone];
                    return;
                }
                break;
            case CriticalDamageCard.CommNoise:
                if (action.Kind == GameActionKind.ChangeSpeed) ship.SetSpeed(action.Speed, game.Topology.Ships[a.Defender].CatalogEntry.Definition.MaximumSpeed);
                else if (action.Kind == GameActionKind.ChooseCommand) game.State.Commands[a.Defender].ReplaceTop(action.Command);
                break;
            case CriticalDamageCard.StructuralDamage:
                ship.DealFacedownDamageCard(Deck(game, a.Defender).Draw());
                EngineEffects.NotifyDamageApplied(game, a.Attacker, a.Defender, GamePhase.ResolveCritical); break;
        }
        ship.FlipCriticalFacedown(card); a.PendingDamageCard = null; ContinueHullDamage(game, a);
    }

    private void ContinueHullDamage(GameContext game, AttackState a)
    {
        if (a.PendingHullCards > 0 && !game.IsDestroyed(a.Defender)) game.Flow.Phase = GamePhase.ChooseDamageCard;
        else CompleteAttack(game, a);
    }

    private void CompleteAttack(GameContext game, AttackState a)
    {
        if (a.MineSpatialIndex >= 0) { CompleteMineAttack(game, a); return; }
        if (a.Counter && !game.IsShip(a.Attacker) && !a.EffectFlags.ContainsKey("ti99.counter-completed"))
        { a.EffectFlags["ti99.counter-completed"] = 1; SquadronAbilityEngine.ResolveTi99CounterActivation(game.SquadronViews[game.SquadronIndex(a.Attacker)]); }
        if (a.IgnitionAttack) IgnitionAttackGeometry.ConsumeTarget(game, a.Attacker);
        if (!a.EnvironmentalDamage)
        {
            OfficialAssaultObjectives.ObserveAttack(game, a); OfficialNavigationObjectives.ObserveAttack(game, a);
            OfficialDefenseObjectives.ObserveAttack(game, a);
            if (!a.AfterAttackWindowEntered)
            {
                a.AfterAttackWindowEntered = true; a.Window = CombatWindow.AfterAttack;
                effects.EnterWindow(game, a, CombatWindow.AfterAttack);
            }
            if (effects.HasPendingChoice(game, a)) { game.Flow.Phase = GamePhase.ModifyAttack; return; }
            if (effects.TryFinishWindow(game, a, CombatWindow.AfterAttack))
            { if (ReferenceEquals(game.Flow.Attacks.LastOrDefault(), a)) game.Flow.Phase = GamePhase.ModifyAttack; return; }
            a.ResponseStarted = true;
            if (TryResponse(game, a)) return;
            if (!game.IsShip(a.Defender) && a.EffectFlags.ContainsKey("defender.spent-token") && !a.EffectFlags.ContainsKey("whisper.resolved"))
            {
                a.EffectFlags["whisper.resolved"] = 1;
                if (MissingSquadronLifecycleEffects.BeginWhisperAfterAttack(game, a.Defender, GamePhase.ModifyAttack))
                { a.EffectFlags["completion.resume"] = 1; return; }
            }
        }
        if (a.EnvironmentalDamage && game.IsShip(a.Defender) && !a.EffectFlags.ContainsKey("objective.obstacle.scored")
            && a.DamageRule is "asteroid-overlap" or "debris-overlap" or "obstacle.purrgil" or "obstacle.exogorth" or "damaged-controls.obstacle")
        {
            a.EffectFlags["objective.obstacle.scored"] = 1;
            OfficialNavigationObjectives.ObserveObstacleDamage(game, a.Defender, a.AppliedDamage, a.HullCardsDealt);
        }
        game.Flow.Attacks.RemoveAt(game.Flow.Attacks.Count - 1);
        if (a.EffectFlags.ContainsKey("granted-attack")) game.Flow.Flags["combat.granted.completed"] = 1;
        if (game.Flow.Attacks.Count > 0)
        {
            if (a.EnvironmentalDamage)
            {
                game.Flow.Phase = a.ResumePhase;
                if (Current(game).AfterAttackWindowEntered && !Current(game).ResponseStarted && Current(game).EffectRule.Length == 0) CompleteAttack(game, Current(game));
                return;
            }
            if (a.EffectFlags.ContainsKey("granted-attack"))
            { game.Flow.Phase = a.ResumePhase; if (!a.EffectFlags.ContainsKey("granted-flak")) ClearGrantedFlags(game); return; }
            // A response finishes after its parent damage has already resolved.
            CompleteAttack(game, Current(game)); return;
        }
        if (!a.EnvironmentalDamage && !a.Counter && !a.IsSalvoAttack && !a.EffectFlags.ContainsKey("granted-attack"))
        {
            if (game.IsShip(a.Attacker) && game.IsShip(a.Defender))
            {
                var activation = game.Flow.ShipActivations[a.Attacker]; activation.AttacksResolved++;
                activation.UsedHullZones |= 1 << (int)a.FromZone;
                activation.AttackedShips = [..activation.AttackedShips, a.Defender];
            }
            else if (a.Attacker >= 0 && !game.IsShip(a.Attacker)) SquadronModule.NotifyAttackCompleted(game);
        }
        game.Flow.Phase = a.ResumePhase;
        if (a.EffectFlags.ContainsKey("granted-attack") && !a.EffectFlags.ContainsKey("granted-flak")) ClearGrantedFlags(game);
        if (!a.EnvironmentalDamage || !game.Flow.Flags.ContainsKey("movement.damage-batch")) RoundModule.CheckTerminal(game);
    }

    private bool TryResponse(GameContext game, AttackState original)
    {
        if (original.Counter || original.IsSalvoAttack || original.SnipeAttack || original.Attacker < 0 || game.IsDestroyed(original.Attacker)) return false;
        int blue = 0; DiceBattery battery; bool salvo = false, quadLaser = false;
        HullZone responseZone = original.ToZone;
        if (!game.IsShip(original.Defender) && !game.IsShip(original.Attacker)
            && !original.EffectFlags.ContainsKey("response.counter.resolved"))
        {
            original.EffectFlags["response.counter.resolved"] = 1;
            var source = game.SquadronViews[game.SquadronIndex(original.Defender)];
            var target = game.SquadronViews[game.SquadronIndex(original.Attacker)];
            blue = SquadronAbilityEngine.GetCounterValue(source, target, game.SceneSquadronViews);
            if (blue <= 0 || !SquadronAbilityEngine.CanCounterAttack(source, target, game.SceneSquadronViews)) return false;
            battery = new(0, blue, 0);
        }
        else if (game.IsShip(original.Defender) && original.Salvo && !game.IsDestroyed(original.Defender)
            && !original.EffectFlags.ContainsKey("response.salvo.resolved"))
        {
            original.EffectFlags["response.salvo.resolved"] = 1;
            var def = game.Topology.Ships[original.Defender].CatalogEntry.Definition;
            var printed = game.IsShip(original.Attacker) ? def.GetBattery(HullZone.Rear) : def.AntiSquadronArmament;
            battery = new(printed.RedDice, original.Range == RangeBand.Long ? 0 : printed.BlueDice, original.Range == RangeBand.Close ? printed.BlackDice : 0);
            salvo = true;
        }
        else if (game.IsShip(original.Defender) && !game.IsShip(original.Attacker)
            && original.EffectFlags.TryGetValue("response.qlt.zone", out long selectedZone)
            && !original.EffectFlags.ContainsKey("response.qlt.resolved"))
        {
            original.EffectFlags["response.qlt.resolved"] = 1;
            var printed = TurbolaserRules.EffectiveFlakArmament(game.ShipViews[original.Defender], game.SquadronViews[game.SquadronIndex(original.Attacker)],
                OffensiveRetrofitRules.EffectiveAntiSquadronArmament(game.ShipViews[original.Defender], new(0, 1, 0)));
            battery = new(printed.RedDice, printed.BlueDice, printed.BlackDice); responseZone = (HullZone)selectedZone; quadLaser = true;
        }
        else return false;
        if (battery.Total == 0) return false;
        var response = new AttackState { Attacker = original.Defender, Defender = original.Attacker,
            FromZone = responseZone, ToZone = original.FromZone, Range = original.Range,
            Counter = !salvo, IsSalvoAttack = salvo, Dice = Unrolled(battery),
            RedDice = battery.Red, BlueDice = battery.Blue, BlackDice = battery.Black,
            ResumePhase = original.ResumePhase, MandatoryRemovals = original.Obstructed ? 1 : 0 };
        if (!salvo && !game.IsShip(response.Attacker))
        {
            var source = game.SquadronViews[game.SquadronIndex(response.Attacker)]; var target = game.SquadronViews[game.SquadronIndex(response.Defender)];
            response.Obstructed = !CloneWarsSquadronRules.AttacksCannotBeObstructed(source)
                && (SquadronAbilityEngine.TreatsDefenderAsObstructed(target) || TitleRules.StrongholdSource(target) is not null
                    || SquadronUpgradeRules.JammingFieldSource(source, target) is not null || CloneWarsSquadronRules.LuminaraObstructs(source, target, game.SceneSquadronViews));
            response.MandatoryRemovals = response.Obstructed ? 1 : 0;
        }
        if (quadLaser)
        {
            TryAttackGeometry(game, response.Attacker, response.Defender, response.FromZone, response.ToZone, out var counterRange, out bool counterObstructed, out _);
            response.Range = counterRange; response.Obstructed = counterObstructed; response.MandatoryRemovals = counterObstructed ? 1 : 0;
            response.EffectFlags["qlt.counter"] = 1;
        }
        game.Flow.Attacks.Add(response); Enter(game, response, GamePhase.ModifyAttack, CombatWindow.BeforeRoll);
        return true;
    }

    private void Enter(GameContext game, AttackState attack, GamePhase phase, CombatWindow window)
    {
        game.Flow.Phase = phase; attack.Window = window;
        if (window == CombatWindow.BeforeRoll && !attack.EffectFlags.ContainsKey("objective.declared"))
        {
            attack.EffectFlags["objective.declared"] = 1; OfficialAssaultObjectives.OnAttackDeclared(game, attack);
            int jamming = OfficialDefenseObjectives.ExtraPreRollRemovals(game, attack);
            attack.MandatoryRemovals += jamming;
            if (jamming > 0) attack.EffectFlags["preroll.jamming.remaining"] = jamming;
            if (OfficialObjectives.Is(game, "opening-salvo") && game.IsShip(attack.Attacker) && game.IsShip(attack.Defender)
                && OfficialObjectives.TokenCount(game, attack.Attacker) > 0)
            { OfficialObjectives.SetTokens(game, attack.Attacker, 0); attack.EffectFlags["objective.opening.pending"] = 1; }
        }
        effects.EnterWindow(game, attack, window);
        if (window == CombatWindow.BeforeRoll) OfficialObjectives.QueueChoices(game,
            OfficialObjectives.TimingChoices(game, "attack-declared", attack.Attacker, attack.Defender), phase);
    }
    private void AddEffectOptions(GameContext game, AttackState a, List<GameActionOption> options)
    {
        options.AddRange(effects.GetActions(game, a, a.Window).Where(option => !IsConcentrateFireUpgrade(option.Template) ||
            a.EffectFlags.ContainsKey(ConcentrateFireStarted) || HasConcentrateFireGrant(game, a)));
    }
    private void ValidateUpgrades(GameContext game, int attacker, int defender)
    {
        foreach (var card in game.UpgradesFor(attacker).Concat(game.UpgradesFor(defender)).Where(card => card.Active))
            if (!BuiltinUpgrades.Contains(card.Definition.id) && !effects.SupportsUpgrade(card.Definition.id)
                && !EngineEffects.SupportsUpgrade(card.Definition.id))
                throw new NotSupportedException("Combat timing for upgrade requires an explicit effect handler: " + card.Definition.id);
    }
    private static AttackState Current(GameContext game) => game.Flow.Attacks.Count > 0 ? game.Flow.Attacks[^1]
        : throw new InvalidOperationException("No active attack continuation.");
    private static int Owner(GameContext game, int entity) => entity < 0 ? OfficialObjectives.SecondPlayer(game) : game.State.Owners[entity];
    private static DamageDeckState Deck(GameContext game, int defender) => game.State.Decks[Owner(game, defender) - 1];
    private static GameActionOption Option(GameAction action, string label) => new(action, label);
    private void RequireOffered(GameContext game, GameAction action)
    {
        // The public entry point validates state-specific choices including
        // physical token slots; arbitrary action IDs cannot mutate combat state.
        var decision = GetDecision(game);
        if (!decision.Actions.Any(option => Equivalent(option.Template, action)))
            throw new ArgumentException("Action is not legal in the current combat decision.", nameof(action));
    }
    private static bool Equivalent(GameAction left, GameAction right) =>
        left with { Dice = null, DiceIndices = null, Course = null } == right with { Dice = null, DiceIndices = null, Course = null }
        && (left.DiceIndices ?? []).SequenceEqual(right.DiceIndices ?? []);
    private static DieFace[] Faces(AttackDieColor color) => color switch
    { AttackDieColor.Red => RedFaces, AttackDieColor.Blue => BlueFaces, AttackDieColor.Black => BlackFaces,
        _ => throw new ArgumentException("Unsupported die color.") };
    private static AttackDie[] Unrolled(DiceBattery battery) => Enumerable.Repeat(AttackDieColor.Red, battery.Red)
        .Concat(Enumerable.Repeat(AttackDieColor.Blue, battery.Blue)).Concat(Enumerable.Repeat(AttackDieColor.Black, battery.Black))
        .Select(color => new AttackDie(color, DieFace.Blank)).ToArray();
    private static HullZoneGeometry Zone(GameContext game, int ship, HullZone zone) => game.State.Ships[ship].HasAuxiliaryHullZones
        ? HullGeometry.Huge(game.ShipBase(ship), zone, game.Topology.Ships[ship].CatalogEntry.Definition.ShipType == "prototype-super-star-destroyer")
        : HullGeometry.Standard(game.ShipBase(ship), zone);
    private static bool ArcIntersectsCircle(HullZoneGeometry zone, Circle circle)
        => DistanceToRay(zone.ArcOrigin, zone.FirstBoundary, circle.Center) <= circle.Radius
            || DistanceToRay(zone.ArcOrigin, zone.SecondBoundary, circle.Center) <= circle.Radius;
    private static float DistanceToRay(Vector2 origin, Vector2 direction, Vector2 point)
        => Vector2.Distance(point, origin + direction * Math.Max(0, Vector2.Dot(point - origin, direction)));
    private static bool FriendlyUpgrade(GameContext game, int entity, string id)
        => game.Topology.Ships.Any(ship => !game.IsDestroyed(ship.EntityIndex) && !game.IsReserved(ship.EntityIndex)
            && Owner(game, ship.EntityIndex) == Owner(game, entity) && game.HasUpgrade(ship.EntityIndex, id));
    private static bool CanDiscardExtraEvade(GameContext game, AttackState a)
        => game.IsShip(a.Attacker) && game.IsShip(a.Defender) && a.SpentEvade.HasValue
            && OffensiveRetrofitEffects.SizeRank(game.Topology.Ships[a.Attacker].CatalogEntry.Metadata.Size)
                > OffensiveRetrofitEffects.SizeRank(game.Topology.Ships[a.Defender].CatalogEntry.Metadata.Size);
    private static IEnumerable<int[]> Subsets(int count, int maximum)
    {
        for (int size = 1; size <= maximum; size++)
        {
            var indices = Enumerable.Range(0, size).ToArray();
            while (true)
            {
                yield return indices.ToArray();
                int i = size - 1;
                while (i >= 0 && indices[i] == count - size + i) i--;
                if (i < 0) break;
                indices[i]++;
                for (int j = i + 1; j < size; j++) indices[j] = indices[j - 1] + 1;
            }
        }
    }
    public static int Keyword(GameContext game, int entity, string keyword)
    {
        if (game.IsShip(entity)) return 0;
        var def = game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition;
        foreach (string entry in def.Keywords)
        {
            string[] words = entry.Replace(':', ' ').Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (words.Length > 0 && string.Equals(words[0], keyword, StringComparison.OrdinalIgnoreCase))
                return words.Length > 1 && int.TryParse(words[^1], out int value) ? value : 1;
        }
        return 0;
    }
    private static IEnumerable<int> EngagingEnemies(GameContext game, int entity) => game.Topology.Squadrons
        .Where(s => s.EntityIndex != entity && game.IsOnBoard(s.EntityIndex)
            && Owner(game, s.EntityIndex) != Owner(game, entity) && Keyword(game, s.EntityIndex, "HEAVY") == 0
            && Geometry2D.Distance(game.SquadronBase(entity), game.SquadronBase(s.EntityIndex)) <= TabletopMeasurements.FromWorldUnits(AttackResolver.SquadronEngagementDistance))
        .Select(s => s.EntityIndex);
    private static bool Engaged(GameContext game, int entity) => game.IsOnBoard(entity) && EngagingEnemies(game, entity).Any();
    private static string FlakKey(GameContext game, int ship) => game.Flow.Flags.ContainsKey("combat.granted.flak")
        ? $"combat.granted.flak-sequence.{ship}" : $"combat.flak.{game.State.Round}.{ship}";
    private static string FlakTargetKey(GameContext game, int ship, int target, HullZone? zone = null)
        => FlakKey(game, ship) + ".zone." + (int)(zone ?? FlakZone(game, ship) ?? HullZone.Front) + ".target." + target;
    private static string FlakBatchTargetKey(GameContext game, int ship, int target) => FlakKey(game, ship)
        + ".batch." + game.Flow.Flags.GetValueOrDefault(FlakKey(game, ship) + ".sequence") + ".target." + target;
    private static bool CanReuseZone(GameContext game, int ship, HullZone zone) => OfficialAssaultObjectives.AllowsZoneReuse(game, ship) || game.HasUpgrade(ship, "gunnery-team")
        && game.Flow.ShipActivations[ship].AttacksResolved == 1 && (game.Flow.ShipActivations[ship].UsedHullZones & (1 << (int)zone)) != 0
        && (game.Flow.ShipActivations[ship].ResolvedCommands & (1 << (int)ShipCommand.ConcentrateFire)) != 0;
    private static HullZone? FlakZone(GameContext game, int ship) => game.Flow.Flags.TryGetValue(FlakKey(game, ship), out long zone) ? (HullZone)(zone - 1) : null;
    private static void ClearGrantedFlags(GameContext game)
    {
        game.Flow.Flags.Remove("combat.granted.active"); game.Flow.Flags.Remove("combat.granted.return"); game.Flow.Flags.Remove("combat.granted.blue");
        game.Flow.Flags.Remove("combat.granted.black"); game.Flow.Flags.Remove("combat.granted.flak"); game.Flow.Flags.Remove("combat.granted.maximum-range");
        game.Flow.Flags.Remove("combat.granted.single-target"); game.Flow.Flags.Remove("combat.granted.upgrade-index");
        game.Flow.Flags.Remove("combat.granted.exhaust"); game.Flow.Flags.Remove("combat.granted.paid");
        game.Flow.Flags.Remove("combat.granted.obstructed"); game.Flow.Flags.Remove("combat.granted.ignore-engagement");
        foreach (string key in game.Flow.Flags.Keys.Where(key => key.StartsWith("combat.granted.flak-sequence.", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
    }
}
