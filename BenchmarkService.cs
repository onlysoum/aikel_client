using Microsoft.Extensions.Configuration;
using Npgsql;
using Pgvector;
using System.Collections.Concurrent;
using System.Diagnostics;

/// <summary>
/// Benchmarks pgvector nearest-neighbour search on the bioclinicbert table
/// using two indexing strategies: HNSW and IVFFlat.
///
/// The benchmark runs against each strategy separately and prints a
/// side-by-side comparison so you can see which index performs better.
///
/// HNSW  – Hierarchical Navigable Small World. Fast queries, higher memory.
/// IVFFlat – Inverted File Index. Lower memory, slightly slower queries.
/// </summary>
public class BenchmarkService
{
    private readonly NpgsqlDataSource _dataSource;
    private readonly BenchmarkConfig  _config;

    // Dimension of BioClinicalBERT embeddings.
    private const int VectorDimension = 768;

    // The two index types we benchmark against.
    private static readonly (string Label, string IndexType)[] IndexStrategies =
    [
        ("HNSW",    "hnsw"),
        ("IVFFlat", "ivfflat"),
    ];

    // ----------------------------------------------------------------
    // Constructor injection
    // ----------------------------------------------------------------
    public BenchmarkService(NpgsqlDataSource dataSource, IConfiguration configuration)
    {
        _dataSource = dataSource;

        _config = new BenchmarkConfig
        {
            ConcurrencyLevel = configuration.GetValue("Benchmark:ConcurrencyLevel", 10),
            Duration         = configuration.GetValue<int?>("Benchmark:DurationSeconds", null) is int s
                                   ? TimeSpan.FromSeconds(s)
                                   : null,
        };
    }

    /// <summary>
    /// Entry point called from Program.cs via the CLI "benchmark" command.
    /// Runs the benchmark for each indexing strategy and prints a comparison.
    /// </summary>
    public async Task RunAsync(int iters)
    {
        var config = _config with { TotalRequests = iters };

        Console.WriteLine("═══════════════════════════════════════════════════════");
        Console.WriteLine("  Aikel Client — Vector Search Benchmark");
        Console.WriteLine("  Comparing: HNSW vs IVFFlat");
        Console.WriteLine("═══════════════════════════════════════════════════════");
        Console.WriteLine($"  Requests      : {config.TotalRequests}");
        Console.WriteLine($"  Concurrency   : {config.ConcurrencyLevel}");
        Console.WriteLine($"  Vector dim    : {VectorDimension}");
        Console.WriteLine($"  Duration limit: {(config.Duration.HasValue ? config.Duration.Value.ToString() : "none")}");
        Console.WriteLine("═══════════════════════════════════════════════════════\n");

        // Warm up the connection pool once before all runs.
        await WarmUpAsync();

        // Pre-generate random unit vectors outside the hot loop.
        Console.WriteLine("  Pre-generating query vectors...");
        var rng        = new Random(42);
        var vectorPool = config.QueryPool
                               .Select(_ => GenerateRandomUnitVector(rng, VectorDimension))
                               .ToArray();
        Console.WriteLine("  Vectors ready.\n");

        // Run benchmark for each index strategy and collect summaries.
        var summaries = new List<(string Label, BenchmarkSummary Summary)>();

        foreach (var (label, indexType) in IndexStrategies)
        {
            Console.WriteLine($"  ── Running benchmark for: {label} ──────────────────");
            var summary = await RunStrategyAsync(config, vectorPool, label, indexType);
            summaries.Add((label, summary));
            Console.WriteLine();
        }

        // Print side-by-side comparison.
        PrintComparison(summaries);
    }

    // ----------------------------------------------------------------
    // Run the full benchmark loop for one indexing strategy.
    // ----------------------------------------------------------------
    private async Task<BenchmarkSummary> RunStrategyAsync(
        BenchmarkConfig config,
        Vector[] vectorPool,
        string label,
        string indexType)
    {
        var results   = new ConcurrentBag<QueryResult>();
        var semaphore = new SemaphoreSlim(config.ConcurrencyLevel, config.ConcurrencyLevel);

        using var cts = config.Duration.HasValue
            ? new CancellationTokenSource(config.Duration.Value)
            : new CancellationTokenSource();

        var totalTimer = Stopwatch.StartNew();
        var tasks      = new List<Task>(config.TotalRequests);

        for (int i = 0; i < config.TotalRequests; i++)
        {
            if (cts.Token.IsCancellationRequested) break;

            var vector = vectorPool[i % vectorPool.Length];

            await semaphore.WaitAsync(cts.Token).ConfigureAwait(false);

            tasks.Add(Task.Run(async () =>
            {
                try
                {
                    var result = await ExecuteSingleQueryAsync(vector, indexType, cts.Token)
                                     .ConfigureAwait(false);
                    results.Add(result);

                    if (results.Count % 10 == 0)
                        Console.WriteLine($"    [{label}] Progress: {results.Count}/{config.TotalRequests} completed");
                }
                finally
                {
                    semaphore.Release();
                }
            }, cts.Token));
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);
        totalTimer.Stop();

        return BenchmarkSummary.From(results.ToList(), totalTimer.Elapsed.TotalMilliseconds);
    }

    // ----------------------------------------------------------------
    // Execute one pgvector search query using the specified index type.
    //
    // pgvector uses the index automatically based on the operator:
    //   <=>  cosine distance  (used by both HNSW and IVFFlat)
    //
    // We set the search parameters via SET commands to ensure the
    // correct index type is used for each run:
    //   hnsw.ef_search   – controls HNSW search quality/speed tradeoff
    //   ivfflat.probes   – controls IVFFlat search quality/speed tradeoff
    // ----------------------------------------------------------------
    private async Task<QueryResult> ExecuteSingleQueryAsync(
        Vector vector, string indexType, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            await using var conn = await _dataSource.OpenConnectionAsync(ct)
                                                    .ConfigureAwait(false);

            // Set index-specific search parameters.
            await using (var setCmd = conn.CreateCommand())
            {
                setCmd.CommandText = indexType switch
                {
                    "hnsw"    => "SET hnsw.ef_search = 64;",
                    "ivfflat" => "SET ivfflat.probes = 10;",
                    _         => "SELECT 1;"
                };
                await setCmd.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
            }

            await using var cmd = conn.CreateCommand();

            // Cosine-distance nearest-neighbour search on bioclinicbert.
            cmd.CommandText = @"
                SELECT id, content, embedding <=> $1 AS distance
                FROM bioclinicbert
                ORDER BY distance
                LIMIT 5;";

            cmd.Parameters.AddWithValue(vector);

            await using var reader = await cmd.ExecuteReaderAsync(ct).ConfigureAwait(false);

            // Consume all rows to measure full result-set transfer time.
            while (await reader.ReadAsync(ct).ConfigureAwait(false)) { }

            sw.Stop();
            return new QueryResult(Success: true, ElapsedMs: sw.ElapsedMilliseconds);
        }
        catch (OperationCanceledException)
        {
            sw.Stop();
            return new QueryResult(Success: false, ElapsedMs: sw.ElapsedMilliseconds,
                                   ErrorMessage: "Cancelled (duration limit reached)");
        }
        catch (Exception ex)
        {
            sw.Stop();
            return new QueryResult(Success: false, ElapsedMs: sw.ElapsedMilliseconds,
                                   ErrorMessage: ex.Message);
        }
    }

    // ----------------------------------------------------------------
    // Generate a random unit vector of the given dimension.
    // ----------------------------------------------------------------
    private static Vector GenerateRandomUnitVector(Random rng, int dimension)
    {
        var values = new float[dimension];
        double sumSq = 0;

        for (int i = 0; i < dimension; i++)
        {
            values[i] = (float)(rng.NextDouble() * 2 - 1);
            sumSq += values[i] * values[i];
        }

        float norm = (float)Math.Sqrt(sumSq);
        for (int i = 0; i < dimension; i++)
            values[i] /= norm;

        return new Vector(values);
    }

    // ----------------------------------------------------------------
    // Warm-up: prime the connection pool before timing starts.
    // ----------------------------------------------------------------
    private async Task WarmUpAsync()
    {
        Console.WriteLine("  Warming up connection pool...");
        try
        {
            await using var conn = await _dataSource.OpenConnectionAsync().ConfigureAwait(false);
            await using var cmd  = conn.CreateCommand();
            cmd.CommandText = "SELECT 1";
            await cmd.ExecuteScalarAsync().ConfigureAwait(false);
            Console.WriteLine("  Warm-up complete.\n");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Warm-up failed (proceeding anyway): {ex.Message}\n");
        }
    }

    // ----------------------------------------------------------------
    // Print a side-by-side comparison of all strategies.
    // ----------------------------------------------------------------
    private static void PrintComparison(List<(string Label, BenchmarkSummary Summary)> results)
    {
        Console.WriteLine("═══════════════════════════════════════════════════════");
        Console.WriteLine("  BENCHMARK COMPARISON RESULTS");
        Console.WriteLine("═══════════════════════════════════════════════════════");

        // Header row.
        Console.WriteLine($"  {"Metric",-25} {"HNSW",12} {"IVFFlat",12}");
        Console.WriteLine($"  {new string('-', 25)} {new string('-', 12)} {new string('-', 12)}");

        var h = results.FirstOrDefault(r => r.Label == "HNSW").Summary;
        var v = results.FirstOrDefault(r => r.Label == "IVFFlat").Summary;

        if (h == null || v == null)
        {
            Console.WriteLine("  Could not compare — one or both strategies failed.");
            return;
        }

        Console.WriteLine($"  {"Total Requests",-25} {h.TotalRequests,12} {v.TotalRequests,12}");
        Console.WriteLine($"  {"Successful",-25} {h.SuccessfulRequests,12} {v.SuccessfulRequests,12}");
        Console.WriteLine($"  {"Failed",-25} {h.FailedRequests,12} {v.FailedRequests,12}");
        Console.WriteLine();
        Console.WriteLine($"  {"Total Time (ms)",-25} {h.TotalElapsedMs,12:F1} {v.TotalElapsedMs,12:F1}");
        Console.WriteLine($"  {"Requests/Second",-25} {h.RequestsPerSecond,12:F2} {v.RequestsPerSecond,12:F2}");
        Console.WriteLine();
        Console.WriteLine($"  {"Avg Latency (ms)",-25} {h.AvgLatencyMs,12:F2} {v.AvgLatencyMs,12:F2}");
        Console.WriteLine($"  {"Min Latency (ms)",-25} {h.MinLatencyMs,12:F2} {v.MinLatencyMs,12:F2}");
        Console.WriteLine($"  {"Max Latency (ms)",-25} {h.MaxLatencyMs,12:F2} {v.MaxLatencyMs,12:F2}");
        Console.WriteLine($"  {"p95 Latency (ms)",-25} {h.P95LatencyMs,12:F2} {v.P95LatencyMs,12:F2}");
        Console.WriteLine($"  {"p99 Latency (ms)",-25} {h.P99LatencyMs,12:F2} {v.P99LatencyMs,12:F2}");

        Console.WriteLine();

        // Winner summary.
        string winner = h.AvgLatencyMs < v.AvgLatencyMs ? "HNSW" : "IVFFlat";
        Console.WriteLine($"  ✔ Lower avg latency: {winner}");

        string throughputWinner = h.RequestsPerSecond > v.RequestsPerSecond ? "HNSW" : "IVFFlat";
        Console.WriteLine($"  ✔ Higher throughput: {throughputWinner}");

        Console.WriteLine("═══════════════════════════════════════════════════════\n");
    }
}
