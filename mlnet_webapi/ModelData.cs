using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace mlnet_webapi
{
    public class ModelData : IDisposable
    {
        public ModelData(byte[] dataBytes)
        {
            this.DataBytes = dataBytes;
        }

        public byte[] DataBytes { get; }

        #region IDisposable Support
        private bool disposedValue = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    
                }

                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }
        #endregion
    }
}
