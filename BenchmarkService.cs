using Microsoft.Extensions.Configuration;
using Npgsql;
using Pgvector;
using System.Collections.Concurrent;
using System.Diagnostics;

/// <summary>
/// Benchmarks pgvector nearest-neighbour search on the bioclinicbert table.
///
/// Instead of calling an external embedding API (which would skew latency
/// numbers), we generate random unit vectors of the correct dimension before
/// the benchmark starts. This isolates pure DB / index performance.
/// </summary>
public class BenchmarkService
{
    private readonly NpgsqlDataSource _dataSource;
    private readonly BenchmarkConfig  _config;

    // Dimension of the vectors stored in the bioclinicbert table.
    // BioClinicalBERT produces 768-dimensional embeddings.
    private const int VectorDimension = 768;

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
    /// </summary>
    public async Task RunAsync(int iters)
    {
        var config = _config with { TotalRequests = iters };

        Console.WriteLine("═══════════════════════════════════════════════════════");
        Console.WriteLine("  Aikel Client — Vector Search Benchmark");
        Console.WriteLine("═══════════════════════════════════════════════════════");
        Console.WriteLine($"  Requests      : {config.TotalRequests}");
        Console.WriteLine($"  Concurrency   : {config.ConcurrencyLevel}");
        Console.WriteLine($"  Vector dim    : {VectorDimension}");
        Console.WriteLine($"  Duration limit: {(config.Duration.HasValue ? config.Duration.Value.ToString() : "none")}");
        Console.WriteLine("═══════════════════════════════════════════════════════\n");

        // Warm up the connection pool.
        await WarmUpAsync();

        // Pre-generate one random unit vector per query in the pool.
        // We do this outside the hot loop so allocation doesn't skew timings.
        Console.WriteLine("  Pre-generating query vectors...");
        var rng        = new Random(42); // fixed seed for reproducibility
        var vectorPool = config.QueryPool
                               .Select(_ => GenerateRandomUnitVector(rng, VectorDimension))
                               .ToArray();
        Console.WriteLine("  Vectors ready.\n");

        // ── Main benchmark loop ─────────────────────────────────────
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
                    var result = await ExecuteSingleQueryAsync(vector, cts.Token)
                                     .ConfigureAwait(false);
                    results.Add(result);

                    if (results.Count % 10 == 0)
                        Console.WriteLine($"  Progress: {results.Count}/{config.TotalRequests} completed");
                }
                finally
                {
                    semaphore.Release();
                }
            }, cts.Token));
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);
        totalTimer.Stop();
        // ── End of benchmark loop ───────────────────────────────────

        var summary = BenchmarkSummary.From(results.ToList(), totalTimer.Elapsed.TotalMilliseconds);
        PrintSummary(summary);
    }

    // ----------------------------------------------------------------
    // Execute one pgvector cosine-distance search query.
    // ----------------------------------------------------------------
    private async Task<QueryResult> ExecuteSingleQueryAsync(
        Vector vector, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            await using var conn = await _dataSource.OpenConnectionAsync(ct)
                                                    .ConfigureAwait(false);

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
    // Unit vectors are standard inputs for cosine-distance search.
    // ----------------------------------------------------------------
    private static Vector GenerateRandomUnitVector(Random rng, int dimension)
    {
        var values = new float[dimension];
        double sumSq = 0;

        for (int i = 0; i < dimension; i++)
        {
            values[i] = (float)(rng.NextDouble() * 2 - 1); // range [-1, 1]
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
    // Console reporter.
    // ----------------------------------------------------------------
    private static void PrintSummary(BenchmarkSummary s)
    {
        Console.WriteLine("\n═══════════════════════════════════════════════════════");
        Console.WriteLine("  BENCHMARK RESULTS");
        Console.WriteLine("═══════════════════════════════════════════════════════");

        Console.WriteLine($"  Total Requests      : {s.TotalRequests}");
        Console.WriteLine($"  Successful          : {s.SuccessfulRequests}");
        Console.WriteLine($"  Failed              : {s.FailedRequests}");

        Console.WriteLine();
        Console.WriteLine($"  Total Time          : {s.TotalElapsedMs:F1} ms  ({s.TotalElapsedMs / 1000.0:F2} s)");
        Console.WriteLine($"  Requests / Second   : {s.RequestsPerSecond:F2} rps");

        Console.WriteLine();
        Console.WriteLine($"  Avg Latency         : {s.AvgLatencyMs:F2} ms");
        Console.WriteLine($"  Min Latency         : {s.MinLatencyMs:F2} ms");
        Console.WriteLine($"  Max Latency         : {s.MaxLatencyMs:F2} ms");
        Console.WriteLine($"  p95 Latency         : {s.P95LatencyMs:F2} ms");
        Console.WriteLine($"  p99 Latency         : {s.P99LatencyMs:F2} ms");

        Console.WriteLine("═══════════════════════════════════════════════════════\n");
    }
}
