# -*- coding: utf-8 -*-
"""fast instance creation"""


class FastInstance:
    __slots__ = []
    __default__ = {}
    __display__ = []

    def __init__(self, *args, **kwargs):
        for idx, attr in enumerate(self.__slots__):
            if idx < len(args):
                value = args[idx]
            elif attr in kwargs:
                value = kwargs[attr]
            elif attr in self.__default__:
                value = self.__default__[attr]
            else:
                raise KeyError(f'missing {attr} for {self.__class__.__name__}')
            setattr(self, attr, value)

    def __repr__(self):
        r = f"[{self.__class__.__name__}"
        if self.__display__:
            idt = [f'{attr}={getattr(self, attr)}'
                   for attr in self.__display__]
            r += f" - {', '.join(idt)}"
        r += "]"
        return r


if __name__ == '__main__':
    pass
