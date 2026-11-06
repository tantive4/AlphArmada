#nullable disable
using System.Collections.Generic;

namespace Armada.Session;

public sealed class SessionValidationResult
{
	private readonly List<string> errors = new List<string>();

	public bool IsValid => errors.Count == 0;

	public IReadOnlyList<string> Errors => errors;

	internal void Add(string message)
	{
		if (!string.IsNullOrWhiteSpace(message))
		{
			errors.Add(message);
		}
	}
}
