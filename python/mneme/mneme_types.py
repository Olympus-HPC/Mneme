from ctypes import Structure, c_uint


class dim3(Structure):
    _fields_ = [("x", c_uint), ("y", c_uint), ("z", c_uint)]

    def __init__(self, x=1, y=1, z=1):
        super().__init__(x, y, z)

    def __repr__(self):
        return f"dim3({self.x}, {self.y}, {self.z})"

    def to_dict(self):
        return {"x": int(self.x), "y": int(self.y), "z": int(self.z)}
