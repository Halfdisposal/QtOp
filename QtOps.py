import numpy as np

class QVecs:
    def __init__(self):
        self.data = None 
    
    def __repr__(self):
        return f"{self.data}"
    
    def __add__(self, x):
        return self.__class__(self.data + x.data)
    
    def __sub__(self, x):
        return self.__class__(self.data - x.data)
    
    def __mul__(self, x):
        if isinstance(x, (int, float, complex)):
            return self.__class__(self.data * x)
        raise TypeError("Use @ for linear algebra multiplication, * is scalar multiplication.")
    
    def __truediv__(self, x):
        if isinstance(x, (int, float, complex)):
            return self.__class__(self.data / x)
        raise TypeError("Division only supported by scalars.")
    
    def __pow__(self, x: int|float):
        return self.__class__(self.data**x)

    def __eq__(self, x):
        return np.allclose(self.data, x.data)
    
    def CT(self):
        return self.data.conjugate().T



class Ket(QVecs):
    def __init__(self, items: list | np.ndarray):
        super().__init__()
        arr = np.array(items, dtype=np.complex128)
        self.data = arr.reshape(-1, 1)

    def dagger(self):
        """Return the conjugate transpose (Bra)."""
        return Bra(self.data.conjugate().T)

    
    def __matmul__(self, x):
        if isinstance(x, Bra):
            return np.outer(self.data, x.data)
        else:
            return 
    
    def CT(self):
        return Bra(self.data.flatten())



class Bra(QVecs):
    def __init__(self, items: list | np.ndarray):
        super().__init__()
        arr = np.array(items, dtype=np.complex128)
        self.data = arr.reshape(1, -1)

    def dagger(self):
        """Return the conjugate transpose (Ket)."""
        return Ket(self.data.conjugate().T)
    

    def __matmul__(self, x):
        if isinstance(x, Ket):
            return np.sum(self.data * x.data)
        else:
            return 
        
    def CT(self):
        return Ket(self.data.flatten())

class QOp:
    def __init__(self):
        self.data: np.ndarray = None 
    
    def __eq__(self, x):
        return np.all(self.data == x.data)

    def __repr__(self):
        return f"{self.data}"
    
    def __add__(self, x):
        return Op(self.data + x.data)
    
    def __sub__(self, x):
        return Op(self.data - x.data)
    
    def __matmul__(self, x):
        """Quantum style multiplication using @"""
        if isinstance(x, Ket):
            return Ket(self.data @ x.data)
        elif isinstance(x, Bra):
            return Bra(x.data @ self.data)
        elif isinstance(x, Op):
            return Op(self.data @ x.data)
        else:
            raise TypeError("Unsupported @ operation")
    
    def __mul__(self, x):
        """Scalar multiplication"""
        if isinstance(x, (int, float, complex)):
            return Op(self.data * x)
        raise TypeError("Use @ for operator application, * is scalar only.")
    
    def ishermitian(self):
        return np.allclose(self.data, self.data.conjugate().T)
    
    def dagger(self):
        return Op(self.data.conjugate().T)


class Op(QOp):
    def __init__(self, item: list[list] | np.ndarray):
        super().__init__()
        arr = np.array(item, dtype=np.complex128)
        self.data = arr
    
    def isunitary(self):
        x = self@self.dagger()
        i = Op(np.identity(x.data.shape[0], dtype=np.complex128))
        return x == i


def commute(A: Op, B: Op):
    return np.allclose((A@B).data, (B@A).data)

def commutator(A: Op, B: Op):
    return A@B - B@A

