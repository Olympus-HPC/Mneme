import json
import shutil
import tempfile
import warnings
import os
from pathlib import Path
from typing import Union, List, Optional
from mneme.llvm.module import parse_assembly

class AlternateIRRecordDB:
    """
    Context manager for creating a temporary record database with alternate LLVM IR.
    
    This class creates a temporary copy of a Mneme record database (JSON file)
    and modifies it to point to a new LLVM IR module. This is useful for
    replaying a recorded kernel with modified IR code.
    
    Usage:
        with AlternateIRRecordDB(record_db="path/to/kernel.json", new_ir="path/to/new.ll") as rdb:
            executor = AsyncReplayExecutor(
                record_db=rdb.path,
                record_id=rdb.rids[0],
                ...
            )
    """
    def __init__(self, record_db: Union[str, Path], new_ir: Union[str, Path]):
        """
        Initialize the AlternateIRRecordDB context manager.
        
        Args:
            record_db: Path to the existing record database JSON file (or directory containing one).
            new_ir: Path to an LLVM IR file (.ll or .bc) or a string containing LLVM IR.
        """
        self.record_db = Path(record_db).absolute()
        self.new_ir = new_ir
        self.temp_dir: Optional[str] = None
        self.path: Optional[str] = None
        self.rids: List[str] = []
        
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mneme_alternate_ir_")
        
        if not self.record_db.exists():
            raise FileNotFoundError(f"Record DB path not found: {self.record_db}")
        
        # A shortcut to pass DBs with only a single kernel in them
        target_json = self.record_db
        if self.record_db.is_dir():
            json_files = list(self.record_db.glob("*.json"))
            if len(json_files) == 1:
                target_json = json_files[0]
            elif len(json_files) == 0:
                raise ValueError(f"No JSON files found in {self.record_db}")
            else:
                raise ValueError(f"Multiple JSON files found in {self.record_db}. Please specify the JSON file directly.")
                 
        with open(target_json, 'r') as f:
            data = json.load(f)
            
        new_bc_path = Path(self.temp_dir) / "alternate_kernel.bc"
        
        # Handle new_ir -- either
        # (1) a str containing LLVM IR
        # (2) a Path to a .ll file
        # (3) a Path to a .bc file
        ir_is_content = False
        if isinstance(self.new_ir, str):
            if os.path.exists(self.new_ir):
                ir_is_content = False
            else:
                # if a string and not a valid file, assume IR
                ir_is_content = True
        elif isinstance(self.new_ir, Path):
            if self.new_ir.exists():
                ir_is_content = False
            else:
                raise FileNotFoundError(f"IR file not found: {self.new_ir}")
        
        if ir_is_content:
            # parse assembly string and write bitcode
            ir_str = str(self.new_ir)
            mod = parse_assembly(ir_str)
            try:
                mod.to_bitcode(str(new_bc_path))
            finally:
                mod._dispose()
        else:
            ir_path = Path(self.new_ir)
            if ir_path.suffix == '.bc':
                shutil.copy(ir_path, new_bc_path)
            else:
                # assume .ll, read and convert
                with open(ir_path, 'r') as f:
                    ir_content = f.read()
                mod = parse_assembly(ir_content)
                try:
                    mod.to_bitcode(str(new_bc_path))
                finally:
                    mod._dispose()

        # update modules in existing record db
        existing_modules = data.get("Modules", [])
        if len(existing_modules) > 1:
            warnings.warn(f"Original record DB had multiple modules: {existing_modules}. Replacing all with single alternate IR.")
             
        data["Modules"] = [str(new_bc_path)]
        
        # populate rids
        self.rids = list(data.get("instances", {}).keys())
        
        # write new JSON
        new_json_path = Path(self.temp_dir) / target_json.name
        with open(new_json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        self.path = str(new_json_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
