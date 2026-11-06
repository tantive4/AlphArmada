using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Setup upgrades and objective instructions share the normal owner-ordered
/// timing window. An offer is selected before its potentially mutating generator runs.</summary>
public static class SetupTimingEffects
{
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        if (timing is not ("setup-before" or "setup-after" or "setup-before-round")) yield break;
        for (int index = 0; index < game.Flow.Continuations.Count; index++)
        {
            var frame = game.Flow.Continuations[index];
            if (!frame.Rule.StartsWith("setup.", StringComparison.Ordinal) || (frame.Rule == "setup.ozzel") != (timing == "setup-before-round")
                || game.IsReserved(frame.Source) || game.IsDestroyed(frame.Source) || !SetupModule.HasSourceChoices(game, frame)) continue;
            string id = "setup.order.source." + frame.Target + "." + frame.Rule + "." + (frame.Indices is { Length: > 0 } ? frame.Indices[0] : -1);
            string name = game.State.Upgrades[frame.Target].Definition.name;
            yield return new(new(GameActionKind.UseAbility, frame.Source, Index: index, AbilityId: id), "Resolve " + name + " setup effect");
        }
        if (timing != "setup-before-round" && HasObjectiveInstruction(game, timing == "setup-after"))
        {
            int owner = OfficialObjectives.Id(game) is "advanced-gunnery" ? OfficialObjectives.FirstPlayer(game) : OfficialObjectives.SecondPlayer(game);
            int anchor = game.Topology.Ships.First(s => !s.IsScenarioUnit && s.Owner == owner).EntityIndex;
            yield return new(new(GameActionKind.UseAbility, anchor, AbilityId: "setup.order.objective"), "Resolve " + OfficialObjectives.Id(game) + " setup instructions");
        }
    }
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (!action.AbilityId.StartsWith("setup.order.", StringComparison.Ordinal)) return false;
        if (!GetActions(game, window.Text, window.Source, window.Target).Any(o => GameSession.Matches(o, action)))
            throw new ArgumentException("This setup timing instruction is unavailable.");
        if (action.AbilityId == "setup.order.objective")
        {
            bool after = window.Text == "setup-after";
            OfficialObjectives.QueueChoices(game, OfficialAssaultObjectives.SetupChoices(game, after)
                .Concat(OfficialNavigationObjectives.SetupChoices(game, after)).Concat(OfficialDefenseObjectives.SetupChoices(game, after))
                .Concat(OfficialCreatureObjectives.SetupChoices(game, after)), GamePhase.AbilityChoice);
        }
        else { game.Flow.Flags["setup.timing.selected"] = action.Index + 1; game.Flow.Phase = GamePhase.Deployment; }
        return true;
    }
    private static bool HasObjectiveInstruction(GameContext game, bool after) => after
        ? OfficialObjectives.Id(game) is ("advanced-gunnery" or "most-wanted" or "opening-salvo" or "precision-strike" or "surprise-attack"
            or "abandoned-mining-facility" or "fleet-in-being" or "fighter-ambush" or "jamming-barrier" or "rift-ambush" or "salvage-run" or "solar-corona")
            || OfficialObjectives.Is(game, "hyperspace-migration") && !MapMovementRules.Enabled(game)
        : OfficialObjectives.Id(game) is "targeting-beacons" or "capture-the-vip" or "fire-lanes" or "hyperspace-assault" or "fighter-ambush"
            or "planetary-ion-cannon" or "dangerous-territory" or "infested-fields" or "intel-sweep" or "minefields" or "salvage-run" or "sensor-net" or "hyperspace-migration";
}

public sealed partial class SetupModule
{
    internal static bool HasSourceChoices(GameContext game, GameContinuation frame) => ChoiceOptions(game, frame).Count > 0;
    private static bool TryBeginSetupTiming(GameContext game, int stage)
    {
        if (game.Flow.Flags.ContainsKey("setup.timing.selected")) return false;
        string[] timings = stage == 0 ? ["setup-before"] : ["setup-after", "setup-before-round"];
        foreach (string timing in timings)
        {
            string flag = "setup.timing.started." + timing;
            if (game.Flow.Flags.ContainsKey(flag)) continue;
            game.Flow.Flags[flag] = 1;
            if (EngineEffects.BeginWindow(game, timing, -1, resumePhase: GamePhase.Deployment)) return true;
        }
        return false;
    }
}
