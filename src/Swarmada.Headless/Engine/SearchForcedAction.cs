using Swarmada.Headless.Engine;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Search;

/// <summary>Fast path for an authoritative, fully determined real action.
/// A single parameterized native offer is NOT a forced action: the finite
/// catalogue must contain exactly one complete legal payload. This helper is
/// deliberately outside simulated hidden-command hypothesis handling.</summary>
public static class ForcedActionAdvancer
{
    public static bool TryApply(GameSession game, IReadOnlyList<SearchKnowledge> knowledge,
        out GameAction? action, out string[] selectedPath, ActionCatalogProfile? profile = null,
        GameSession? beforeWorkspace = null, GameDecision? settledDecision = null)
    {
        ArgumentNullException.ThrowIfNull(game); ArgumentNullException.ThrowIfNull(knowledge);
        action = null; selectedPath = [];
        // Most ordinary windows expose several fixed, already legal payloads.
        // Proving that from their exact native offers avoids constructing the
        // finite catalogue here and again when the meaningful search starts.
        var native = settledDecision ?? game.GetDecision();
        if (native.Kind != DecisionKind.Player || CanProveMultipleFixedActions(native, profile?.IncludeUiNavigation ?? false))
            return false;
        var adapter = new DecisionAdapter(game, profile);
        var decision = adapter.Current;
        if (decision.Kind != DecisionKind.Player || adapter.AtomicActionCount != 1) return false;
        if (ReferenceEquals(game, beforeWorkspace)) throw new ArgumentException("Before-state workspace must be independent.", nameof(beforeWorkspace));
        var before = beforeWorkspace ?? game.DeepClone();
        if (beforeWorkspace is not null) before.CopyFrom(game);
        var path = new List<string>();
        for (int prefix = 0; prefix < 256; prefix++)
        {
            decision = adapter.Current;
            if (decision.Kind != DecisionKind.Player || decision.Choices.Count != 1)
                throw new InvalidOperationException("A singleton complete catalogue produced an ambiguous conditional choice.");
            string id = decision.Choices[0].Id; path.Add(id);
            if (!adapter.Select(id)) continue;
            action = adapter.LastCommittedAction ?? throw new InvalidOperationException("Forced action did not expose its native payload.");
            game.GetDecision();
            foreach (var observer in knowledge) observer.ObserveTransition(before, action, game);
            selectedPath = path.ToArray(); return true;
        }
        throw new InvalidOperationException("A forced conditional action did not terminate within its explicit prefix bound.");
    }

    /// <summary>Negative singleton test only. False means the finite catalogue
    /// is still required. Exact duplicates and excluded UI navigation do not
    /// count; a single parameterized offer can still contain many legal actions.</summary>
    public static bool CanProveMultipleFixedActions(GameDecision decision, bool includeUiNavigation = false)
    {
        ArgumentNullException.ThrowIfNull(decision);
        if (decision.Kind != DecisionKind.Player) return false;
        GameAction? first = null;
        foreach (var offer in decision.Actions)
        {
            if (offer.AcceptsPosition || offer.AcceptsYaw || offer.AcceptsCourse || offer.AcceptsFlip ||
                !includeUiNavigation && offer.Template.AbilityId == "squadron.deselect") continue;
            if (first is null) first = offer.Template;
            else if (!ExactGameActionComparer.Instance.Equals(first, offer.Template)) return true;
        }
        return false;
    }
}
