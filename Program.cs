using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Npgsql;
using System.CommandLine;

HostApplicationBuilder builder = Host.CreateApplicationBuilder(args);
/*
 * Benchmark Service
 * Search Service
 * CRUD Service (maybe)
 * Logging extensions 
 */

string connectionString = builder.Configuration.GetConnectionString("DefaultConnection")
    ?? "Host=localhost;Database=postgres;Username=postgres;Password=postgres";

builder.Services.AddTransient<BenchmarkService>();
builder.Services.AddTransient<SearchService>();
builder.Services.AddSingleton(NpgsqlDataSource.Create(connectionString));

IHost host = builder.Build();


var rootCmd = new RootCommand("Aikel Client");

// Benchmark
var iterationsOption = new Option<int>(name: "--iterations", ["-i"]) {
    Description = "The number of iterations.",
    Required = true,
};

var benchmarkCommand = new Command("benchmark", "Run performance tests on vector indexes") { iterationsOption };

benchmarkCommand.SetAction(async (iters) => {
    var service = host.Services.GetRequiredService<BenchmarkService>();
    await service.RunAsync(iters.GetValue(iterationsOption));
});

// Search
var queryOption = new Option<string>(name: "--query", ["-q"]) {
    Description = "The query to search the db for",
    Required = true
};
var searchCommand = new Command("search", "Do a vector search with the DB for a query") { queryOption };

searchCommand.SetAction(async (query) => {
    var service = host.Services.GetRequiredService<SearchService>();
    await service.RunAsync(query.GetValue(queryOption) ?? "not given");
});

rootCmd.Add(benchmarkCommand);
rootCmd.Add(searchCommand);

return await rootCmd.Parse(args).InvokeAsync();
 