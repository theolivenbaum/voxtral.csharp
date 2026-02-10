using System;

namespace Voxtral
{
    public interface IVoxtralModel : IDisposable
    {
        string Transcribe(string wavPath);
    }
}
