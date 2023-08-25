import numpy as np


class ArrayWrapper(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, getitem, setitem):
        self.__getitem = getitem
        self.__setitem = setitem

    # array indexing (main reason for the ArrayWrapper class)
    def __getitem__(self, key):
        return self.__getitem(key)

    def __setitem__(self, key, val):
        return self.__setitem(key, val)

    # for print out in console
    def __repr__(self):
        return repr(self.__getitem__(...))

    def __array__(self, dtype=None):
        array = self.__getitem__(...)
        if dtype is None:
            return array
        else:
            return array.astype(dtype=dtype)

    # def __len__(self):
    #     return len(self.__array__())

    # all the other stuff
    def __getattr__(self, name):
        return getattr(self.__getitem__(...), name)


class PropertyForArray(property):
    def __get__(self, instance, owner):
        def getitem(key):
            return self.fget(instance, key)

        def setitem(key, val):
            # print(self, instance, owner)
            assert self.fset is not None, "a setter must also be defined!"
            self.fset(instance, val, key)  # in Pulse class key is a kwarg so comes last

        return ArrayWrapper(getitem, setitem)


class Test:
    def __init__(self, array):
        self._x = array

    @PropertyForArray
    def x(self, key=...):
        return self._x[key]

    @x.setter
    def x(self, val, key=...):
        self._x[key] = val


t = Test(np.arange(10))
t.x[1] = 1
