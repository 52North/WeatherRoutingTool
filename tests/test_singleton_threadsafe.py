import threading
import time

from wrt_singleton import SingletonBase


class DummySingleton(metaclass=SingletonBase):
    def __init__(self):
        # small delay to increase chance of concurrent construction
        time.sleep(0.01)


class OtherSingleton(metaclass=SingletonBase):
    def __init__(self):
        time.sleep(0.005)


def _create(instances, idx, cls):
    instances[idx] = cls()


def test_singleton_threadsafe_single_class():
    n_threads = 50
    instances = [None] * n_threads
    threads = []

    for i in range(n_threads):
        t = threading.Thread(target=_create, args=(instances, i, DummySingleton))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # all entries should be the same object
    ids = {id(x) for x in instances}
    assert len(ids) == 1


def test_singleton_threadsafe_different_classes():
    a = DummySingleton()
    b = OtherSingleton()
    assert a is not b
