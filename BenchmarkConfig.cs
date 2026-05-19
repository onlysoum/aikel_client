public record BenchmarkConfig
{
    public int TotalRequests { get; init; } = 100;
    public int ConcurrencyLevel { get; init; } = 10;
    public TimeSpan? Duration { get; init; } = null;
    public IReadOnlyList<string> QueryPool { get; init; } =
    [
        "machine learning model training",
        "neural network architecture",
        "natural language processing",
        "vector database indexing",
        "semantic similarity search",
        "transformer attention mechanism",
        "embedding space representation",
        "cosine distance metric",
        "approximate nearest neighbour",
        "dimensionality reduction techniques"
    ];
}
