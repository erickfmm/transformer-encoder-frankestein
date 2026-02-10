import os
import logging
import tempfile
from pathlib import Path

# ==================== STORAGE MANAGER ====================
class StorageManager:
    """Manages disk usage to stay under 500GB limit"""
    
    def __init__(self, limit_gb: float = 500.0):
        self.limit_bytes = limit_gb * 1024**3
        self.used_bytes = 0
        self.temp_files = []
        
    def register_file(self, path: str) -> bool:
        """Register a file and check if within limits"""
        try:
            size = os.path.getsize(path)
            self.used_bytes += size
            
            if self.used_bytes > self.limit_bytes:
                logging.warning(f"Storage limit exceeded: {self.used_bytes/1024**3:.2f}GB")
                return False
            return True
        except:
            return True
    
    def create_temp_file(self, suffix: str = ".tmp") -> str:
        """Create a temporary file and register it"""
        temp_dir = Path("./temp_data")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(
            dir=temp_dir, 
            suffix=suffix, 
            delete=False
        )
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def cleanup(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except:
                pass
        self.temp_files.clear()
