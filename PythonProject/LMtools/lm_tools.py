import os


class OrderedSetList:
    """
    OrderedSetList 类结合了 list 和 set 的特性，旨在提供一个有序且唯一的元素集合。
    它维护元素的插入顺序，同时确保集合中的所有元素都是唯一的。
    """

    def __init__(self):
        """
        初始化 OrderedSetList 对象。
        self._list 用于维护元素的插入顺序。
        self._set 用于快速检查元素是否存在。
        """
        self._list = []
        self._set = set()

    def append(self, item):
        """
        如果 item 不在集合中，则将其添加到集合和列表中。
        这样可以确保所有元素都是唯一的，并且维持插入顺序。
        """
        if item not in self._set:
            self._list.append(item)
            self._set.add(item)

    def extend(self, items):
        """
        将多个元素添加到集合中。
        通过调用 append 方法，自动去重并保持顺序。
        """
        for item in items:
            self.append(item)

    def __contains__(self, item):
        """
        检查集合中是否存在指定的元素。
        通过 self._set 快速判断元素是否存在，提高查询效率。
        """
        return item in self._set

    def __len__(self):
        """
        返回集合中元素的数量。
        由于 self._list 维护了元素顺序，所以通过它来确定元素数量。
        """
        return len(self._list)

    def __getitem__(self, index):
        """
        支持通过索引访问集合中的元素。
        该方法使得 OrderedSetList 类的对象可以像列表一样被索引。
        """
        return self._list[index]

    def __iter__(self):
        """
        返回集合的迭代器。
        该方法使得 OrderedSetList 类的对象可以被迭代。
        """
        return iter(self._list)

    def __repr__(self):
        """
        返回集合的字符串表示。
        该方法主要用于调试和日志记录，提供对象的详细信息。
        """
        return f"{self.__class__.__name__}({self._list})"

    def remove(self, item):
        """
        如果元素存在于集合中，则从集合和列表中移除该元素。
        这样做可以确保元素的唯一性和有序性得到维护。
        """
        if item in self._set:
            self._list.remove(item)
            self._set.remove(item)

    def discard(self, item):
        """
        如果元素存在于集合中，则移除它，如果不存在则什么也不做。
        与 remove 方法不同的是，discard 在元素不存在时不抛出异常。
        """
        if item in self._set:
            self._list.remove(item)
            self._set.discard(item)

    def to_list(self):
        """
        将 OrderedSetList 转换为普通列表。
        该方法提供了一种方式来获取集合的副本，作为常规列表使用。
        """
        return self._list.copy()
