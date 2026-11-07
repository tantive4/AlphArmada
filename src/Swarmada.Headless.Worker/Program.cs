using System.Text.Json;
using Swarmada.Headless.Worker;

if (args.Contains("--help", StringComparer.Ordinal) || args.Length == 0)
{
    Console.WriteLine("Usage: dotnet Swarmada.Headless.Worker.dll run [--config FILE] [--data DIR] [--output DIR] [--run-id ID] [--worker-id ID] [--cell-id ID] [--games N (0=continuous)] [--preset small|mixed] [--objective ID] [--match FILE] [--setup FILE] [--model FILE] [--simulations 200|600] [--concurrency N] [--cpu-workers N] [--batch-size N] [--batch-delay-ms N] [--max-game-seconds N] [--max-steps N] [--duration-seconds N] [--drain-seconds N] [--memory-limit-mb N] [--replay] [--validate-only]");
    return 0;
}
try
{
    var options = WorkerOptions.Parse(args);
    using var log = new WorkerLog(options); using var lifetime = new WorkerLifetime(options, log);
    return await new WorkerRunner(options, log, lifetime).RunAsync();
}
catch (Exception error)
{
    Console.Error.WriteLine(JsonSerializer.Serialize(new { eventType = "fatal", timestampUtc = DateTimeOffset.UtcNow, error = error.ToString() }, WorkerOptions.Json));
    return 1;
}
