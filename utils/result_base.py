# -*- coding: utf-8 -*-
"""result base class"""


class ResultBase:
    __name__ = 'ResultBase'
    __slots__ = []
    __display__ = []

    def __init__(self, *args, **kwargs):
        for idx, attr in enumerate(self.__slots__):
            if idx < len(args):
                value = args[idx]
            elif attr in kwargs:
                value = kwargs[attr]
            else:
                raise KeyError(f'missing {attr} for {self.__name__}')
            setattr(self, attr, value)

    def __repr__(self):
        idt = [f'{attr}={getattr(self, attr)}'
               for attr in self.__display__]
        return f"[{self.__name__} - {', '.join(idt)}]"


if __name__ == '__main__':
    pass
