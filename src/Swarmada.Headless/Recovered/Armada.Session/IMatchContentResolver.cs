#nullable disable
namespace Armada.Session;

public interface IMatchContentResolver
{
	bool TryResolveShip(string definitionId, out MatchUnitValidationRules rules);

	bool TryResolveSquadron(string definitionId, out MatchUnitValidationRules rules);
}
