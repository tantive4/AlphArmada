using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Explicit, nestable timing windows. Choices and return addresses are snapshot data.</summary>
public static class EngineEffects
{
    public static void CleanupRound(GameContext game, int previousRound)
    {
        // Transient flags must not grow with elapsed rounds or increase every later
        // search snapshot indefinitely, including custom scenarios without a round limit.
        string[] transientPrefixes = ["movement.", "repair.", "combat.", "activation.", "relay.", "effects.used."];
        foreach (var key in game.Flow.Flags.Keys.Where(key => transientPrefixes.Any(prefix => key.StartsWith(prefix, StringComparison.Ordinal))).ToArray())
            game.Flow.Flags.Remove(key);
    }
    public static bool SupportsUpgrade(string id) => GeneralAbilityEffects.SupportsUpgrade(id)
        || MovementAbilityEffects.SupportsUpgrade(id) || RoundAbilityEffects.SupportsUpgrade(id) || ActivationAbilityEffects.SupportsUpgrade(id)
        || MapAbilityEffects.SupportsUpgrade(id) || DamageEvents.SupportsUpgrade(id) || TokenAssignments.SupportsUpgrade(id)
        || SetupModule.SupportsUpgrade(id);
    public static bool NotifyDamageApplied(GameContext game, int source, int target, GamePhase returnPhase)
    { DamageEvents.Observe(game, source); return DamageEvents.HasPending(game); }
    public static bool BeginWindow(GameContext game, string timing, int source, int target = -1,
        GamePhase resumePhase = GamePhase.SelectCourse)
    {
        using var scope = new GameScope(game);
        long serial = game.Flow.Flags.GetValueOrDefault("effects.serial") + 1;
        game.Flow.Flags["effects.serial"] = serial;
        var frame = new GameContinuation("effects.window", source, target, (int)resumePhase, serial, timing);
        OfficialObjectives.PrepareTimingWindow(game, frame);
        if (game.SuspendTimingWindow?.Invoke(frame) == true)
        {
            // Search must materialize a private rule input before even generating
            // native offers: a placeholder can otherwise suppress a legal effect.
            game.Flow.Continuations.Add(frame); game.Flow.PendingOptions = [];
            game.Flow.Phase = GamePhase.AbilityChoice; return true;
        }
        var options = GetOptions(game, frame).ToArray();
        bool optional = options.Length > 0;
        if (optional)
        {
            game.Flow.Continuations.Add(frame); game.Flow.PendingOptions = options;
            game.Flow.Phase = GamePhase.AbilityChoice;
        }
        return optional;
    }
    internal static GameDecision GetDecision(GameContext game)
    {
        var frame = Current(game);
        if (frame.Rule != "effects.window") return TokenAssignments.Handles(frame.Rule) ? TokenAssignments.GetDecision(game, frame)
            : DravenAbilityEffects.Handles(frame.Rule) ? DravenAbilityEffects.GetDecision(game, frame)
            : OfficialObjectives.Handles(frame.Rule) ? OfficialObjectives.GetDecision(game, frame)
            : StrategicEffects.Handles(frame.Rule) ? StrategicEffects.GetDecision(game, frame)
            : MissingSquadronLifecycleEffects.Handles(frame.Rule) ? MissingSquadronLifecycleEffects.GetDecision(game, frame)
            : DamageEvents.Handles(frame.Rule) ? DamageEvents.GetDecision(game, frame)
            : MapAbilityEffects.Handles(frame.Rule) ? MapAbilityEffects.GetDecision(game, frame)
            : MovementAbilityEffects.Handles(frame.Rule) ? MovementAbilityEffects.GetDecision(game, frame)
            : frame.Rule.StartsWith("round.", StringComparison.Ordinal) ? RoundAbilityEffects.GetDecision(game, frame) : GeneralAbilityEffects.GetDecision(game, frame);
        if (game.SuspendTimingWindow?.Invoke(frame) == true)
            return new(DecisionKind.Player, game.State.Owners[frame.Source], GamePhase.AbilityChoice, []);
        var options = GetOptions(game, frame).ToList();
        bool mandatory = options.Any(OfficialObjectives.IsMandatoryAction);
        if (!mandatory) options.Add(new(new(GameActionKind.Pass, frame.Source, frame.Target, AbilityId: "effects.finish"), "Finish timing window"));
        int owner = options.Count > 0 && options[0].Template.Source >= 0 ? game.State.Owners[options[0].Template.Source]
            : frame.Source >= 0 ? game.State.Owners[frame.Source] : game.Flow.CurrentPlayer;
        return new(!mandatory && options.Count == 1 ? DecisionKind.Forced : DecisionKind.Player, !mandatory && options.Count == 1 ? 0 : owner,
            GamePhase.AbilityChoice, options);
    }
    internal static void Apply(GameContext game, GameAction action)
    {
        var frame = Current(game);
        if (frame.Rule != "effects.window")
        {
            if (TokenAssignments.Handles(frame.Rule)) TokenAssignments.Apply(game, frame, action);
            else if (DravenAbilityEffects.Handles(frame.Rule)) DravenAbilityEffects.Apply(game, frame, action);
            else if (OfficialObjectives.Handles(frame.Rule)) OfficialObjectives.Apply(game, frame, action);
            else if (StrategicEffects.Handles(frame.Rule)) StrategicEffects.Apply(game, frame, action);
            else if (MissingSquadronLifecycleEffects.Handles(frame.Rule)) MissingSquadronLifecycleEffects.Apply(game, frame, action);
            else if (DamageEvents.Handles(frame.Rule)) DamageEvents.Apply(game, frame, action);
            else if (MapAbilityEffects.Handles(frame.Rule)) MapAbilityEffects.Apply(game, frame, action);
            else if (MovementAbilityEffects.Handles(frame.Rule)) MovementAbilityEffects.Apply(game, frame, action);
            else if (frame.Rule.StartsWith("round.", StringComparison.Ordinal)) RoundAbilityEffects.Apply(game, frame, action);
            else GeneralAbilityEffects.Apply(game, frame, action);
            return;
        }
        if (action.Kind == GameActionKind.Pass && action.AbilityId == "effects.finish")
        {
            var remaining = GetOptions(game, frame).ToArray();
            if (remaining.Any(OfficialObjectives.IsMandatoryAction)) throw new ArgumentException("Mandatory timing effects remain unresolved.");
            if (remaining.Length > 0)
            {
                int owner = game.State.Owners[remaining[0].Template.Source];
                var next = frame with { Indices = (frame.Indices ?? []).Append(owner).ToArray() };
                game.Flow.Continuations[^1] = next;
                if (GetOptions(game, next).Any()) return;
                CloseWindow(game, next); return;
            }
            CloseWindow(game, frame); return;
        }
        if (!GetOptions(game, frame).Any(x => GameSession.Matches(x, action))) throw new ArgumentException("Ability is no longer available.", nameof(action));
        if (SetupTimingEffects.TryApply(game, frame, action) || DravenAbilityEffects.TryApply(game, frame, action) || OfficialObjectives.TryApply(game, frame, action) || StrategicEffects.TryApply(game, frame, action) || MissingSquadronLifecycleEffects.TryApply(game, frame, action) || ActivationAbilityEffects.TryWindowApply(game, frame, action) || MapAbilityEffects.TryApply(game, frame, action) || MovementAbilityEffects.TryApply(game, frame, action)
            || RoundAbilityEffects.TryApply(game, frame, action) || GeneralAbilityEffects.TryApply(game, frame, action))
        { game.Flow.Flags[UsedKey(frame, action)] = 1; return; }
        throw new NotSupportedException("Source ability has not been ported: " + action.AbilityId);
    }
    private static IEnumerable<GameActionOption> GetOptions(GameContext game, GameContinuation window)
    {
        var options = AllOptions(game, window).GroupBy(x => x.Template)
            .Select(x => x.First()).ToArray();
        var passed = window.Indices ?? [];
        int sourceOwner = window.Source >= 0 ? game.State.Owners[window.Source] : 0;
        int owner = options.Where(x => x.Template.Source >= 0 && !passed.Contains(game.State.Owners[x.Template.Source]))
            .Select(x => game.State.Owners[x.Template.Source]).OrderBy(x => sourceOwner == 0 ? x == game.Topology.FirstPlayer ? 0 : 1 : x != sourceOwner ? 0 : 1).FirstOrDefault();
        return options.Where(x => x.Template.Source >= 0 && game.State.Owners[x.Template.Source] == owner);
    }
    private static IEnumerable<GameActionOption> AllOptions(GameContext game, GameContinuation window)
    {
        foreach (var option in SetupTimingEffects.GetActions(game, window.Text, window.Source, window.Target))
            if (!game.Flow.Flags.ContainsKey(UsedKey(window, option.Template))) yield return option;
        foreach (var option in OfficialObjectives.TimingActions(game, window))
            if (!game.Flow.Flags.ContainsKey(UsedKey(window, option.Template))) yield return option;
        if (TryTiming(window.Text, out var timing))
        {
            var sources = game.IsShip(window.Source) ? [window.Source] : window.Source >= 0 ? Array.Empty<int>() :
                game.Topology.Ships.Where(x => !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex))
                    .OrderBy(x => x.Owner == game.Topology.FirstPlayer ? 0 : 1).Select(x => x.EntityIndex).ToArray();
            foreach (int source in sources)
            {
                var activation = game.Flow.ShipActivations[source];
                var context = new AbilityQueryContext(timing, revealedCommand: window.Text == "command-reveal"
                    ? activation.ActualRevealedCommand ?? activation.RevealedCommand : activation.RevealedCommand);
                foreach (var action in UnitAbilityEngine.GetActions(game.ShipViews[source], context))
                {
                    // The recovered Ahsoka offer assumes free capacity before its
                    // discard. The native handler offers the complete exchange and
                    // presents the discarded token as a separate player choice.
                    if (action.AbilityId.StartsWith(GeneralAbilityEffects.AhsokaActionPrefix, StringComparison.Ordinal)) continue;
                    var value = new GameAction(GameActionKind.UseAbility, source, window.Target, AbilityId: action.AbilityId);
                    if (!game.Flow.Flags.ContainsKey(UsedKey(window, value))) yield return new(value, action.Label);
                }
            }
        }
        foreach (var option in MovementAbilityEffects.GetActions(game, window.Text, window.Source, window.Target)
            .Concat(DravenAbilityEffects.GetActions(game, window.Text, window.Source, window.Target))
            .Concat(OfficialObjectives.GetActions(game, window.Text, window.Source, window.Target))
            .Concat(StrategicEffects.GetActions(game, window.Text, window.Source, window.Target))
            .Concat(MissingSquadronLifecycleEffects.GetActions(game, window.Text, window.Source, window.Target))
            .Concat(MapAbilityEffects.GetActions(game, window.Text, window.Source, window.Target))
            .Concat(RoundAbilityEffects.GetActions(game, window.Text, window.Source, window.Target))
            .Concat(GeneralAbilityEffects.GetActions(game, window.Text, window.Source, window.Target)))
            if (!game.Flow.Flags.ContainsKey(UsedKey(window, option.Template))) yield return option;
    }
    private static bool TryTiming(string timing, out AbilityTiming value)
    {
        value = timing switch
        {
            "start-round" => AbilityTiming.StartOfRound, "start-ship-phase" => AbilityTiming.StartOfShipPhase,
            "start-squadron-phase" => AbilityTiming.StartOfSquadronPhase, "start-activation" => AbilityTiming.StartOfActivation,
            "command-reveal" => AbilityTiming.CommandReveal, "during-activation" => AbilityTiming.DuringActivation,
            "determine-course" => AbilityTiming.DetermineCourse, "after-maneuver" => AbilityTiming.AfterManeuver,
            "end-activation" => AbilityTiming.EndOfActivation, "end-phase" => AbilityTiming.EndOfPhase,
            "status" => AbilityTiming.StatusPhase, _ => AbilityTiming.Setup
        };
        return value != AbilityTiming.Setup;
    }
    private static string UsedKey(GameContinuation window, GameAction action)
    {
        // Comms Net's native action IDs encode the selected token and recipient.
        // Those are arguments of one effect, not additional uses of the card.
        // Keep the physical source and event serial in the key so another ship's
        // copy, or this same card at a later trigger, remains independent.
        string effect = action.AbilityId.StartsWith("reveal.comms-net.", StringComparison.Ordinal)
            ? "reveal.comms-net" : action.AbilityId;
        if (action.AbilityId is DravenAbilityEffects.DravenLookAction or DravenAbilityEffects.NoLookAction)
            effect = DravenAbilityEffects.DravenLookAction;
        return $"effects.used.{window.Value}.{action.Source}.{effect}"
            + (action.AbilityId.StartsWith("round.ready-", StringComparison.Ordinal) ? "." + action.Index : "");
    }
    internal static GameContinuation Current(GameContext game) => game.Flow.Continuations.Count > 0 ? game.Flow.Continuations[^1] :
        throw new InvalidOperationException("Ability choice has no continuation.");
    internal static void CloseWindow(GameContext game, GameContinuation window)
    {
        if (Current(game) != window) throw new InvalidOperationException("Ability windows must close in stack order.");
        game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1);
        OfficialObjectives.State(game)?.TimingWindows.Remove(window.Value);
        foreach (var key in game.Flow.Flags.Keys.Where(x => x.StartsWith($"effects.used.{window.Value}.", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
        game.Flow.PendingOptions = []; game.Flow.Phase = (GamePhase)window.Stage;
    }
    public static void PushChoice(GameContext game, GameContinuation continuation, IEnumerable<GameActionOption> options)
    { game.Flow.Continuations.Add(continuation); game.Flow.PendingOptions = options.ToArray(); game.Flow.Phase = GamePhase.AbilityChoice; }
    public static void ReturnFromChoice(GameContext game)
    {
        game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1);
        game.Flow.PendingOptions = []; game.Flow.Phase = GamePhase.AbilityChoice;
    }
}

public sealed class AbilityModule : IGameplayModule
{
    public bool Handles(GamePhase phase) => phase == GamePhase.AbilityChoice;
    public GameDecision GetDecision(GameContext game) => EngineEffects.GetDecision(game);
    public void Apply(GameContext game, GameAction action) => EngineEffects.Apply(game, action);
}
