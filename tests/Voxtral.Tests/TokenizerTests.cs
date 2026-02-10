using System;
using System.IO;
using System.Collections.Generic;
using System.Text.Json;
using Xunit;

namespace Voxtral.Tests
{
    public class TokenizerTests : IDisposable
    {
        private readonly string _modelDir = "test_model";

        public TokenizerTests()
        {
            Directory.CreateDirectory(_modelDir);
        }

        public void Dispose()
        {
            if (Directory.Exists(_modelDir))
            {
                Directory.Delete(_modelDir, true);
            }
        }

        [Fact]
        public void TestDecoding()
        {
            // Create tekken.json
            var data = new
            {
                config = new { default_num_special_tokens = 2 },
                special_tokens = new[]
                {
                    new { rank = 0, token_bytes = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes("[UNK]")) },
                    new { rank = 1, token_bytes = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes("[BOS]")) }
                },
                vocab = new[]
                {
                    new { token_bytes = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes("He")) }, // id 2
                    new { token_bytes = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes("llo")) }, // id 3
                    new { token_bytes = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(" ")) },   // id 4
                    new { token_bytes = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes("World")) }// id 5
                }
            };

            string json = JsonSerializer.Serialize(data);
            File.WriteAllText(Path.Combine(_modelDir, "tekken.json"), json);

            var tokenizer = new Tokenizer(_modelDir);

            // "Hello World" -> 2, 3, 4, 5
            string text = tokenizer.Decode(new[] { 2, 3, 4, 5 });
            Assert.Equal("Hello World", text);

            // Special tokens skipped
            text = tokenizer.Decode(new[] { 0, 1, 2, 3 });
            Assert.Equal("Hello", text);
        }
    }
}
