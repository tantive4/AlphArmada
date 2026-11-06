#nullable disable
using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;

namespace Armada.Session;

public sealed class FleetJsonRepository
{
	private static readonly UTF8Encoding Utf8WithoutBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

	public string RootDirectory { get; }

	public string TrashDirectory => Path.Combine(RootDirectory, "Trash");

	public FleetJsonRepository(string rootDirectory)
	{
		ArgumentException.ThrowIfNullOrWhiteSpace(rootDirectory);
		RootDirectory = Path.GetFullPath(rootDirectory);
	}

	public string Save(FleetListDto fleet)
	{
		EnsureValidFleet(fleet);
		EnsureSafeDocumentId(fleet.fleetId);
		Directory.CreateDirectory(RootDirectory);
		fleet.savedUtc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
		string fleetPath = GetFleetPath(fleet.fleetId);
		string text = Path.Combine(RootDirectory, "fleet-" + fleet.fleetId + "." + Guid.NewGuid().ToString("N") + ".tmp");
		try
		{
			WriteDurably(text, JsonSerializer.Serialize(fleet, new JsonSerializerOptions { IncludeFields = true, WriteIndented = true }));
			ValidateStagedDocument(text);
			if (File.Exists(fleetPath))
			{
				File.Replace(text, fleetPath, null, ignoreMetadataErrors: true);
			}
			else
			{
				File.Move(text, fleetPath);
			}
			return fleetPath;
		}
		finally
		{
			if (File.Exists(text))
			{
				File.Delete(text);
			}
		}
	}

	public FleetListDto Load(string fleetId)
	{
		EnsureSafeDocumentId(fleetId);
		string fleetPath = GetFleetPath(fleetId);
		if (!File.Exists(fleetPath))
		{
			throw new FileNotFoundException("Fleet was not found.", fleetPath);
		}
		return ReadAndValidate(fleetPath);
	}

	public bool TryLoad(string fleetId, out FleetListDto fleet, out string error)
	{
		try
		{
			fleet = Load(fleetId);
			error = string.Empty;
			return true;
		}
		catch (Exception ex)
		{
			fleet = null;
			error = ex.Message;
			return false;
		}
	}

	public FleetRepositoryScanResult ScanAll()
	{
		FleetRepositoryScanResult fleetRepositoryScanResult = new FleetRepositoryScanResult();
		if (!Directory.Exists(RootDirectory))
		{
			return fleetRepositoryScanResult;
		}
		string[] files = Directory.GetFiles(RootDirectory, "*.json", SearchOption.TopDirectoryOnly);
		Array.Sort(files, StringComparer.OrdinalIgnoreCase);
		string[] array = files;
		foreach (string path in array)
		{
			try
			{
				fleetRepositoryScanResult.AddFleet(ReadAndValidate(path));
			}
			catch (Exception ex)
			{
				fleetRepositoryScanResult.AddError(Path.GetFileName(path) + ": " + ex.Message);
			}
		}
		return fleetRepositoryScanResult;
	}

	public bool Delete(string fleetId)
	{
		EnsureSafeDocumentId(fleetId);
		string fleetPath = GetFleetPath(fleetId);
		if (!File.Exists(fleetPath))
		{
			return false;
		}
		File.Delete(fleetPath);
		return true;
	}

	public string MoveToTrash(string fleetId)
	{
		EnsureSafeDocumentId(fleetId);
		string fleetPath = GetFleetPath(fleetId);
		if (!File.Exists(fleetPath))
		{
			throw new FileNotFoundException("Fleet was not found.", fleetPath);
		}
		Directory.CreateDirectory(TrashDirectory);
		string text = DateTime.UtcNow.ToString("yyyyMMdd-HHmmssfff", CultureInfo.InvariantCulture);
		string text2 = Path.Combine(TrashDirectory, "fleet-" + fleetId + "-" + text + ".json");
		File.Move(fleetPath, text2);
		return text2;
	}

	public int CountTrashed()
	{
		if (!Directory.Exists(TrashDirectory))
		{
			return 0;
		}
		return Directory.GetFiles(TrashDirectory, "*.json", SearchOption.TopDirectoryOnly).Length;
	}

	private string GetFleetPath(string fleetId)
	{
		return Path.Combine(RootDirectory, "fleet-" + fleetId + ".json");
	}

	private static FleetListDto ReadAndValidate(string path)
	{
		string json = File.ReadAllText(path, Utf8WithoutBom);
		FleetListDto fleetListDto;
		try
		{
			fleetListDto = JsonSerializer.Deserialize<FleetListDto>(json, new JsonSerializerOptions { IncludeFields = true, MaxDepth = 64 });
		}
		catch (Exception innerException)
		{
			throw new InvalidDataException("Fleet JSON could not be parsed.", innerException);
		}
		EnsureValidFleet(fleetListDto);
		return fleetListDto;
	}

	private static void ValidateStagedDocument(string path)
	{
		ReadAndValidate(path);
	}

	private static void EnsureValidFleet(FleetListDto fleet)
	{
		SessionValidationResult sessionValidationResult = SessionValidation.ValidateFleet(fleet);
		if (sessionValidationResult.IsValid)
		{
			return;
		}
		throw new InvalidDataException("Fleet document is invalid: " + string.Join(" ", sessionValidationResult.Errors));
	}

	private static void EnsureSafeDocumentId(string fleetId)
	{
		if (string.IsNullOrWhiteSpace(fleetId) || fleetId.Length > 128)
		{
			throw new ArgumentException("Fleet ID is missing or too long.", "fleetId");
		}
		foreach (char c in fleetId)
		{
			if ((c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9') && c != '-' && c != '_')
			{
				throw new ArgumentException("Fleet ID contains a character that is unsafe for a document name.", "fleetId");
			}
		}
	}

	private static void WriteDurably(string path, string text)
	{
		using FileStream fileStream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
		using (StreamWriter streamWriter = new StreamWriter(fileStream, Utf8WithoutBom, 4096, leaveOpen: true))
		{
			streamWriter.Write(text);
			streamWriter.Flush();
		}
		fileStream.Flush(flushToDisk: true);
	}
}
