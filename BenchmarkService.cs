using Microsoft.Extensions.Configuration;
using Npgsql;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

/// <summary>
/// Benchmarks pgvector nearest-neighbour search on the bioclinicbert table.
/// Each query is converted to a 1536-dim embedding via OpenAI, then searched
/// against the table using the pgvector <=> (cosine distance) operator.
/// </summary>
public class BenchmarkService
{
    private readonly NpgsqlDataSource _dataSource;
    private readonly BenchmarkConfig  _config;
    private readonly HttpClient       _httpClient;
    private readonly string           _openAiApiKey;

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

        _openAiApiKey = configuration["OpenAI:ApiKey"]
            ?? throw new InvalidOperationException(
                "OpenAI:ApiKey is missing. Add it to appsettings.json or set the OPENAI__APIKEY environment variable.");

        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", _openAiApiKey);
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
        Console.WriteLine($"  Duration limit: {(config.Duration.HasValue ? config.Duration.Value.ToString() : "none")}");
        Console.WriteLine("═══════════════════════════════════════════════════════\n");

        // Warm up the connection pool.
        await WarmUpAsync();

        // Pre-generate embeddings for the query pool to avoid calling OpenAI
        // inside the hot loop, which would skew latency numbers.
        Console.WriteLine("  Generating embeddings for query pool...");
        var embeddingCache = new Dictionary<string, float[]>();
        foreach (var q in config.QueryPool)
        {
            if (!embeddingCache.ContainsKey(q))
                embeddingCache[q] = await GetEmbeddingAsync(q);
        }
        Console.WriteLine("  Embeddings ready.\n");

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

            string query     = config.QueryPool[i % config.QueryPool.Count];
            float[] embedding = embeddingCache[query];

            await semaphore.WaitAsync(cts.Token).ConfigureAwait(false);

            tasks.Add(Task.Run(async () =>
            {
                try
                {
                    var result = await ExecuteSingleQueryAsync(embedding, cts.Token)
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
        float[] embedding, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            await using var conn = await _dataSource.OpenConnectionAsync(ct)
                                                    .ConfigureAwait(false);

            // Enable pgvector type support on this connection.
            conn.TypeMapper.UseVector();

            await using var cmd = conn.CreateCommand();

            // Cosine-distance nearest-neighbour search on bioclinicbert.
            // Adjust column names below if yours differ.
            cmd.CommandText = @"
                SELECT id, content, embedding <=> $1 AS distance
                FROM bioclinicbert
                ORDER BY distance
                LIMIT 5;";

            cmd.Parameters.AddWithValue(new Pgvector.Vector(embedding));

            await using var reader = await cmd.ExecuteReaderAsync(ct).ConfigureAwait(false);

            // Consume all rows so we measure full result-set transfer time.
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
    // Call OpenAI Embeddings API and return the embedding vector.
    // ----------------------------------------------------------------
    private async Task<float[]> GetEmbeddingAsync(string text)
    {
        var payload = JsonSerializer.Serialize(new
        {
            model = "text-embedding-ada-002",
            input = text
        });

        var response = await _httpClient.PostAsync(
            "https://api.openai.com/v1/embeddings",
            new StringContent(payload, Encoding.UTF8, "application/json"))
            .ConfigureAwait(false);

        response.EnsureSuccessStatusCode();

        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var values = doc.RootElement
                        .GetProperty("data")[0]
                        .GetProperty("embedding")
                        .EnumerateArray()
                        .Select(e => e.GetSingle())
                        .ToArray();

        return values;
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
