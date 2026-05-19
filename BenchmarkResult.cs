public record QueryResult(
    bool   Success,
    long   ElapsedMs,
    string? ErrorMessage = null
);

public record BenchmarkSummary
{
    public int    TotalRequests      { get; init; }
    public int    SuccessfulRequests { get; init; }
    public int    FailedRequests     { get; init; }
    public double TotalElapsedMs     { get; init; }
    public double RequestsPerSecond  { get; init; }
    public double AvgLatencyMs       { get; init; }
    public double MinLatencyMs       { get; init; }
    public double MaxLatencyMs       { get; init; }
    public double P95LatencyMs       { get; init; }
    public double P99LatencyMs       { get; init; }

    public static BenchmarkSummary From(
        IReadOnlyList<QueryResult> results,
        double totalElapsedMs)
    {
        var successful = results.Where(r => r.Success).ToList();
        var failed     = results.Where(r => !r.Success).ToList();
        var latencies  = successful.Select(r => (double)r.ElapsedMs)
                                   .OrderBy(x => x).ToArray();

        double avg = latencies.Length > 0 ? latencies.Average() : 0;
        double min = latencies.Length > 0 ? latencies[0]        : 0;
        double max = latencies.Length > 0 ? latencies[^1]       : 0;
        double p95 = Percentile(latencies, 95);
        double p99 = Percentile(latencies, 99);
        double rps = totalElapsedMs > 0 ? results.Count / (totalElapsedMs / 1_000.0) : 0;

        return new BenchmarkSummary
        {
            TotalRequests      = results.Count,
            SuccessfulRequests = successful.Count,
            FailedRequests     = failed.Count,
            TotalElapsedMs     = totalElapsedMs,
            RequestsPerSecond  = rps,
            AvgLatencyMs       = avg,
            MinLatencyMs       = min,
            MaxLatencyMs       = max,
            P95LatencyMs       = p95,
            P99LatencyMs       = p99,
        };
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        if (sorted.Length == 0) return 0;
        int index = (int)Math.Ceiling(percentile / 100.0 * sorted.Length) - 1;
        index = Math.Clamp(index, 0, sorted.Length - 1);
        return sorted[index];
    }
}
