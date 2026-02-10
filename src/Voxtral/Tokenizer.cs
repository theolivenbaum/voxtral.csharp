using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Voxtral
{
    public class Tokenizer
    {
        private readonly Dictionary<int, byte[]> _tokenBytes = new();
        private readonly int _nSpecial;
        private readonly HashSet<int> _specialIds = new();

        public Tokenizer(string modelDir)
        {
            string tekkenPath = Path.Combine(modelDir, "tekken.json");
            string json = File.ReadAllText(tekkenPath);
            var data = JsonSerializer.Deserialize<TekkenData>(json);

            if (data == null) throw new Exception("Failed to load tekken.json");

            _nSpecial = data.Config?.DefaultNumSpecialTokens ?? 1000;

            if (data.SpecialTokens != null)
            {
                foreach (var st in data.SpecialTokens)
                {
                    if (st.Rank.HasValue)
                    {
                        _specialIds.Add(st.Rank.Value);
                    }
                }
            }

            if (data.Vocab != null)
            {
                for (int i = 0; i < data.Vocab.Count; i++)
                {
                    var tokenInfo = data.Vocab[i];
                    if (tokenInfo.TokenBytes != null)
                    {
                        byte[] bytes = Convert.FromBase64String(tokenInfo.TokenBytes);
                        // Token ID = i + nSpecial
                        _tokenBytes[i + _nSpecial] = bytes;
                    }
                }
            }
        }

        public string Decode(IEnumerable<int> tokenIds)
        {
            List<byte> bytes = new List<byte>();
            foreach (var id in tokenIds)
            {
                if (id < _nSpecial || _specialIds.Contains(id))
                {
                    continue;
                }

                if (_tokenBytes.TryGetValue(id, out var b))
                {
                    bytes.AddRange(b);
                }
            }
            // Use simple string decoding. Handling partial UTF-8 sequences is complex (streaming decoder needed),
            // but for full sequence decoding this is fine.
            return Encoding.UTF8.GetString(bytes.ToArray());
        }

        private class TekkenData
        {
            [JsonPropertyName("vocab")]
            public List<VocabItem> Vocab { get; set; }

            [JsonPropertyName("config")]
            public ConfigData Config { get; set; }

            [JsonPropertyName("special_tokens")]
            public List<SpecialToken> SpecialTokens { get; set; }
        }

        private class VocabItem
        {
            [JsonPropertyName("token_bytes")]
            public string TokenBytes { get; set; }
        }

        private class ConfigData
        {
            [JsonPropertyName("default_num_special_tokens")]
            public int DefaultNumSpecialTokens { get; set; }
        }

        private class SpecialToken
        {
            [JsonPropertyName("rank")]
            public int? Rank { get; set; }
        }
    }
}
